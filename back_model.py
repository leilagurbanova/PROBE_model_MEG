from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import count
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


# Helpers

def softmax_probs(q: np.ndarray, beta: float) -> np.ndarray:
    x = beta * q
    x = x - logsumexp(x)
    return np.exp(x)


def apply_lapse(probs: np.ndarray, eps: float) -> np.ndarray:
    n = probs.size
    return (1.0 - eps) * probs + eps * (1.0 / n)


def clip_prob(x: float, lo: float = 1e-12, hi: float = 1.0 - 1e-12) -> float:
    return float(np.clip(x, lo, hi))


def resolve_positive_outcome_idx(
    outcome_values: List[int],
    positive_outcome_value: Optional[int] = None,
    require_explicit: bool = False,
) -> int:
    """Resolve reward-coded outcome index after remapping outcomes to 0..O-1."""
    if len(outcome_values) == 0:
        raise ValueError('outcome_values must be non-empty.')
    if positive_outcome_value is None:
        if require_explicit:
            raise ValueError(
                'positive_outcome_value must be provided explicitly to avoid ambiguous reward coding. '
                f'Observed outcomes: {outcome_values}.'
            )
        return len(outcome_values) - 1
    if positive_outcome_value not in outcome_values:
        raise ValueError(
            f'positive_outcome_value={positive_outcome_value} is not in observed outcomes {outcome_values}.'
        )
    return int(outcome_values.index(int(positive_outcome_value)))


def probe_qc_metrics(out: Dict[str, np.ndarray | float]) -> Dict[str, float | int]:
    """Compute block-level QC metrics for PROBE dynamics."""
    probe_active = np.asarray(out['probe_active'], dtype=float)
    lam_actor = np.asarray(out['lam_actor'], dtype=float)
    n_confirm = int(np.sum(np.asarray(out['confirmation'], dtype=int)))
    n_reject = int(np.sum(np.asarray(out['rejection'], dtype=int)))
    n_switch_in = int(np.sum(np.asarray(out['switch_in'], dtype=int)))
    n_switch_to_alt = int(np.sum(np.asarray(out.get('switch_to_alt', np.zeros(1)), dtype=int)))
    n_switch_to_probe = int(np.sum(np.asarray(out.get('switch_to_probe', np.zeros(1)), dtype=int)))

    probe_active_mean = float(np.mean(probe_active)) if probe_active.size else float('nan')
    p_lam_actor_gt_0_5 = float(np.mean(lam_actor > 0.5)) if lam_actor.size else float('nan')
    degenerate_probe = int(
        probe_active_mean > 0.95
        and p_lam_actor_gt_0_5 < 0.05
        and (n_confirm + n_reject) == 0
    )

    return {
        'probe_active_mean': probe_active_mean,
        'p_lam_actor_gt_0.5': p_lam_actor_gt_0_5,
        'n_confirm': n_confirm,
        'n_reject': n_reject,
        'n_switch_in': n_switch_in,
        'n_switch_to_alt': n_switch_to_alt,
        'n_switch_to_probe': n_switch_to_probe,
        'degenerate_probe': degenerate_probe,
    }



# PROBE model

class Strategy:
    # A strategy TS_m with selective (Q) and predictive (gamma) mappings.
    _uid_counter = count()

    def __init__(self, n_states: int, n_actions: int, n_outcomes: int, uid: Optional[int] = None) -> None:
        self.S = int(n_states)
        self.A = int(n_actions)
        self.O = int(n_outcomes)
        # Stable strategy identity (hypothesis id) used for buffer uniqueness.
        self.uid = int(next(self._uid_counter) if uid is None else uid)
        self.Q = np.zeros((self.S, self.A), dtype=float)
        self.counts = np.ones((self.S, self.A, self.O), dtype=float)
        # Contextual mapping is uniform (no context cues in Donoso task)
        self.kappa = np.ones(self.S, dtype=float) / float(self.S)
        self.last_actor_t = -10**9

    def gamma(self, s: int, a: int) -> np.ndarray:
        c = self.counts[s, a]
        return c / c.sum()

    def update_actor_learning(self, s: int, a: int, o: int, r: float, alpha_q: float) -> None:
        # Selective mapping (Rescorla-Wagner form)
        self.Q[s, a] = alpha_q * r + (1.0 - alpha_q) * self.Q[s, a]
        # Predictive mapping (Dirichlet counts)
        self.counts[s, a, o] += 1.0

    @staticmethod
    def uniform_U(n_states: int, n_actions: int, n_outcomes: int) -> 'Strategy':
        return Strategy(n_states, n_actions, n_outcomes)

    @staticmethod
    def mix_from_LTM(ltm: List['Strategy'], n_states: int, n_actions: int, n_outcomes: int,
                     eta: float) -> 'Strategy':
        # Probe creation: Mp = eta*U + (1-eta)*mean(LTM).
        U = Strategy.uniform_U(n_states, n_actions, n_outcomes)
        p = Strategy(n_states, n_actions, n_outcomes)

        if len(ltm) == 0:
            p.Q = U.Q.copy()
            p.counts = U.counts.copy()
            p.kappa = U.kappa.copy()
            return p

        w = np.ones(len(ltm), dtype=float) / float(len(ltm))

        Q_mix = np.zeros((n_states, n_actions), dtype=float)
        counts_mix = np.zeros((n_states, n_actions, n_outcomes), dtype=float)
        kappa_mix = np.zeros(n_states, dtype=float)

        for wi, st in zip(w, ltm):
            Q_mix += wi * st.Q
            counts_mix += wi * st.counts
            kappa_mix += wi * st.kappa

        p.Q = eta * U.Q + (1.0 - eta) * Q_mix
        p.counts = eta * U.counts + (1.0 - eta) * counts_mix
        p.kappa = eta * U.kappa + (1.0 - eta) * kappa_mix
        return p


@dataclass
class ProbeParams:
    N: int
    tau: float
    eta: float
    theta: float
    alpha_q: float
    beta: float
    eps: float

    kappa_explore: float = 0.0  # Hard threshold for explore gate (Donoso-compatible)
    kappa_switch: float = 0.0   # Hard threshold for switch gate (Donoso-compatible)

    buffer_recency_temp: float = 5.0

    buffer_choice_seed: int = 0


class ProbeEngine:
    def __init__(self, n_states: int, n_actions: int, n_outcomes: int) -> None:
        self.S = int(n_states)
        self.A = int(n_actions)
        self.O = int(n_outcomes)

    @staticmethod
    def _full_normalize(l0: float, l: np.ndarray) -> Tuple[float, np.ndarray]:
        total = l0 + float(l.sum())
        if total <= 0 or not np.isfinite(total):
            n = len(l) + 1
            return 1.0 / n, np.full(len(l), 1.0 / n, dtype=float)
        return l0 / total, l / total

    def run(
        self,
        s: np.ndarray,
        a: np.ndarray,
        o: np.ndarray,
        p: ProbeParams,
        positive_outcome_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray | float]:
        T = len(s)
        reward_idx = (self.O - 1) if positive_outcome_idx is None else int(positive_outcome_idx)
        if reward_idx < 0 or reward_idx >= self.O:
            raise ValueError(
                f'positive_outcome_idx must be in [0, {self.O - 1}], got {reward_idx}.'
            )

        ltm: List[Strategy] = []
        # CLEF-style monitored list: buffer[0] is dummy, monitored non-dummy are buffer[1:].
        buffer: List[Strategy] = []
        actor_idx = 1

        dummy = Strategy.uniform_U(self.S, self.A, self.O)
        init_actor = Strategy.uniform_U(self.S, self.A, self.O)
        buffer.append(dummy)
        buffer.append(init_actor)
        ltm.append(init_actor)

        lam0 = 0.5  # dummy reliability
        lam = np.array([1.0 - lam0], dtype=float)  # reliabilities for buffer[1:]
        lam0, lam = self._full_normalize(lam0, lam)

        ll = 0.0

        lam_actor = np.zeros(T)
        lam_alt1 = np.zeros(T)
        lam_alt2 = np.zeros(T)
        lambda_uncertainty = np.zeros(T)
        lambda_competition = np.zeros(T)
        lambda_entropy = np.zeros(T)
        lambda_entropy_norm = np.zeros(T)
        lambda_K = np.zeros(T)
        lambda_top2_gap = np.zeros(T)
        lam_actor_pre = np.zeros(T)
        lam_alt1_pre = np.zeros(T)
        lam_alt2_pre = np.zeros(T)
        lambda_uncertainty_pre = np.zeros(T)
        lambda_competition_pre = np.zeros(T)
        lambda_entropy_pre = np.zeros(T)
        lambda_entropy_norm_pre = np.zeros(T)
        lambda_K_pre = np.zeros(T)
        lambda_top2_gap_pre = np.zeros(T)
        lam0_pre_trace = np.zeros(T)
        lam_actor_post_cleanup = np.zeros(T)
        lam_alt1_post_cleanup = np.zeros(T)
        lam_alt2_post_cleanup = np.zeros(T)
        lambda_uncertainty_post_cleanup = np.zeros(T)
        lambda_competition_post_cleanup = np.zeros(T)
        lambda_entropy_post_cleanup = np.zeros(T)
        lambda_entropy_norm_post_cleanup = np.zeros(T)
        lambda_K_post_cleanup = np.zeros(T)
        lambda_top2_gap_post_cleanup = np.zeros(T)
        lam0_trace = np.zeros(T)
        q_actor = np.zeros(T)
        gamma_actor = np.zeros(T)
        rpe_actor = np.zeros(T)
        p_explore_trace = np.zeros(T)
        p_switch_trace = np.zeros(T)
        probe_active = np.zeros(T, dtype=int)
        exploration = np.zeros(T, dtype=int)
        switch_in = np.zeros(T, dtype=int)
        switch_to_alt = np.zeros(T, dtype=int)
        switch_to_probe = np.zeros(T, dtype=int)
        reject = np.zeros(T, dtype=int)
        confirm = np.zeros(T, dtype=int)
        n_monitored = np.zeros(T, dtype=int)
        n_ltm = np.zeros(T, dtype=int)

        mode = 'exploit'
        probe_idx: Optional[int] = None
        pending_switch_in = False

        def lambda_prior(n_monitored_before_probe: int) -> float:
            # n_monitored_before_probe: monitored strategies before adding new probe.
            low = 1.0 / (n_monitored_before_probe + 1.0)
            high = 1.0 / 3.0
            # confirmation bias:
            # 1) minimally informative prior for new probe actor (lambda_new)
            # 2) bias it toward reliability threshold 1/2 via theta.
            lam_unbiased = low if low <= high else high
            return p.theta * 0.5 + (1.0 - p.theta) * lam_unbiased

        def lambda_features(
            l0: float,
            l: np.ndarray,
            actor_i: int,
        ) -> Tuple[int, float, float, float, float, float, float, float, float, float]:
            actor_i = int(np.clip(actor_i, 0, len(l) - 1))
            lam_actor_t = float(l[actor_i])
            other = [float(l[i]) for i in range(len(l)) if i != actor_i]
            other_sorted = sorted(other, reverse=True)
            lam_alt1_t = other_sorted[0] if len(other_sorted) >= 1 else 0.0
            lam_alt2_t = other_sorted[1] if len(other_sorted) >= 2 else 0.0
            uncertainty_t = 1.0 - lam_actor_t
            competition_t = lam_alt1_t - lam_actor_t
            top2_gap_t = lam_actor_t - lam_alt1_t
            lam_all = np.concatenate(([float(l0)], l.astype(float)))
            lam_all = np.clip(lam_all, 1e-12, 1.0)
            entropy_t = float(-np.sum(lam_all * np.log(lam_all)))
            K_t = float(len(l) + 1)  # includes lambda0
            denom = np.log(K_t) if K_t > 1.0 else np.nan
            entropy_norm_t = float(np.clip(entropy_t / denom, 0.0, 1.0)) if np.isfinite(denom) and denom > 0 else np.nan
            return (
                actor_i,
                lam_actor_t,
                lam_alt1_t,
                lam_alt2_t,
                uncertainty_t,
                competition_t,
                top2_gap_t,
                entropy_t,
                K_t,
                entropy_norm_t,
            )

        def move_non_dummy_to_end(
            w: np.ndarray,
            move_idx: int,
            w_extra: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
            # CLEF LRU operation: selected monitored strategy becomes most recent.
            if move_idx <= 0 or move_idx >= len(buffer):
                raise RuntimeError(f'Invalid monitored index for move-to-end: {move_idx}')
            st = buffer.pop(move_idx)
            buffer.append(st)
            wi = move_idx - 1  # w indexes buffer[1:]
            wv = float(w[wi])
            w = np.delete(w, wi)
            w = np.append(w, wv)
            if w_extra is not None:
                wv_extra = float(w_extra[wi])
                w_extra = np.delete(w_extra, wi)
                w_extra = np.append(w_extra, wv_extra)
            return w, len(buffer) - 1, w_extra

        def drop_oldest_non_dummy(
            w: np.ndarray,
            cur_actor_idx: int,
            w_extra: Optional[np.ndarray] = None,
        ) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
            # CLEF overflow cleanup: drop oldest monitored strategy (buffer[1]).
            if len(buffer) <= 1:
                raise RuntimeError('Cannot drop non-dummy from an empty monitored list.')
            buffer.pop(1)
            w = np.delete(w, 0)
            if w_extra is not None:
                w_extra = np.delete(w_extra, 0)
            if cur_actor_idx > 1:
                cur_actor_idx -= 1
            elif cur_actor_idx == 1:
                cur_actor_idx = max(1, len(buffer) - 1)
            return w, cur_actor_idx, w_extra

        for t in range(T):
            st = int(s[t])
            at = int(a[t])
            ot = int(o[t])

            # Donoso-style timing: switch-in is detected on feedback at t,
            # but exploration starts on the next stimulus (trial t+1).
            if mode == 'exploit' and pending_switch_in:
                n_monitored_before_probe = max(1, len(buffer) - 1)
                probe = Strategy.mix_from_LTM(ltm, self.S, self.A, self.O, eta=p.eta)
                buffer.append(probe)
                probe_idx = len(buffer) - 1
                prior_probe = lambda_prior(n_monitored_before_probe)
                lam = np.append(lam, prior_probe)
                lam0, lam = self._full_normalize(lam0, lam)
                mode = 'explore'
                pending_switch_in = False

            if mode == 'exploit':
                if actor_idx <= 0 or actor_idx >= len(buffer):
                    actor_idx = len(buffer) - 1
            else:
                if probe_idx is None:
                    raise RuntimeError('Explore mode but no probe present.')
                actor_idx = int(np.clip(probe_idx, 1, len(buffer) - 1))

            if mode == 'explore':
                exploration[t] = 1
                probe_active[t] = 1
                if probe_idx is None:
                    raise RuntimeError('Explore mode but no probe present.')
                actor_idx = probe_idx
            acting_actor_idx = int(actor_idx - 1)  # index in lam/post vectors

            actor = buffer[actor_idx]
            q_actor[t] = float(actor.Q[st, at])
            gamma_actor[t] = float(actor.gamma(st, at)[ot])
            probs_actor = softmax_probs(actor.Q[st], p.beta)
            probs_actor = apply_lapse(probs_actor, p.eps)
            probs_actor = probs_actor / probs_actor.sum()

            if mode == 'exploit':
                # Donoso: in exploitation, actor selects actions (softmax over Q).
                # Alternative strategies are tracked in lambda but do NOT contribute
                # to action log-likelihood. Using soft mixture would change parameter
                # estimates and make model non-comparable to Donoso's original.
                probs = probs_actor
                # Trace p_switch for diagnostics (not used in LL computation).
                actor_wi_now = int(np.clip(actor_idx - 1, 0, len(lam) - 1))
                best_wi_now = int(np.argmax(lam))
                if best_wi_now != actor_wi_now and p.kappa_switch > 0:
                    gap_now = float(lam[best_wi_now] - lam[actor_wi_now])
                    p_switch_ll = float(_sigmoid(gap_now / max(p.kappa_switch, 1e-6)))
                    p_switch_trace[t] = p_switch_ll
                else:
                    p_switch_trace[t] = 0.0
            else:
                probs = probs_actor
                p_explore_trace[t] = 1.0

            ll += np.log(clip_prob(probs[at]))

            gamma0 = 1.0 / self.O
            gammas = np.array([buf.gamma(st, at)[ot] for buf in buffer[1:]], dtype=float)
            # Working copy of likelihoods — tracked through cleanup for hazard transition.
            gammas_work = gammas.copy()
            # Keep previous prior for Donoso-style hazard transition.
            lam_prev0 = float(lam0)
            lam_prev = lam.copy()

            # Feedback-pre lambda traces: prior beliefs before current outcome update.
            (
                _,
                lam_actor_pre[t],
                lam_alt1_pre[t],
                lam_alt2_pre[t],
                lambda_uncertainty_pre[t],
                lambda_competition_pre[t],
                lambda_top2_gap_pre[t],
                lambda_entropy_pre[t],
                lambda_K_pre[t],
                lambda_entropy_norm_pre[t],
            ) = lambda_features(float(lam_prev0), lam_prev, acting_actor_idx)
            lam0_pre_trace[t] = float(lam_prev0)

            # Donoso-style arbitration:
            # 1) posterior responsibilities from model evidences (feedback-time)
            # 2) switch/reject/confirm decisions on posterior
            # 3) hazard transition for next-trial prediction.
            mu0 = gamma0 * lam0
            mu = gammas * lam
            Z = mu0 + float(mu.sum())
            if Z <= 0.0 or not np.isfinite(Z):
                K = len(lam) + 1
                post0 = 1.0 / K
                post = np.full(len(lam), 1.0 / K, dtype=float)
            else:
                post0 = mu0 / Z
                post = mu / Z

            # Feedback-time lambda traces (posterior, before hazard).
            (
                _,
                lam_actor[t],
                lam_alt1[t],
                lam_alt2[t],
                lambda_uncertainty[t],
                lambda_competition[t],
                lambda_top2_gap[t],
                lambda_entropy[t],
                lambda_K[t],
                lambda_entropy_norm[t],
            ) = lambda_features(float(post0), post, acting_actor_idx)
            lam0_trace[t] = float(post0)

            r = 1.0 if ot == reward_idx else 0.0
            rpe_actor[t] = r - q_actor[t]
            actor.update_actor_learning(st, at, ot, r, alpha_q=p.alpha_q)
            actor.last_actor_t = t

            # Donoso-style switch detection at feedback t (posterior).
            # Posterior used for cleanup and transition decisions.
            post0_work = float(post0)
            post_work = post.copy()

            if mode == 'explore':
                assert probe_idx is not None
                probe_wi = probe_idx - 1
                probe_lam = float(post_work[probe_wi])

                cf_indices = [i for i in range(1, len(buffer)) if i != probe_idx]
                cf_best_idx = max(cf_indices, key=lambda i: post_work[i - 1]) if cf_indices else None
                cf_best_lam = float(post_work[cf_best_idx - 1]) if cf_best_idx is not None else 0.0

                if cf_best_lam > 0.5 and probe_lam <= 0.5:
                    reject[t] = 1
                    buffer.pop(probe_idx)
                    post_work = np.delete(post_work, probe_wi)
                    lam_prev = np.delete(lam_prev, probe_wi)
                    gammas_work = np.delete(gammas_work, probe_wi)
                    mode = 'exploit'
                    probe_idx = None
                    if cf_best_idx is None:
                        actor_idx = len(buffer) - 1
                    else:
                        # Account for probe removal before selected counterfactual.
                        if cf_best_idx > probe_wi + 1:
                            cf_best_idx -= 1
                        post_work, actor_idx, lam_prev_aligned = move_non_dummy_to_end(
                            post_work,
                            cf_best_idx,
                            w_extra=lam_prev,
                        )
                        if lam_prev_aligned is None:
                            raise RuntimeError('Failed to align previous lambda prior during reject switch.')
                        lam_prev = lam_prev_aligned
                        # Keep gammas_work aligned with post-cleanup buffer.
                        _gwi = cf_best_idx - 1
                        _gv = float(gammas_work[_gwi])
                        gammas_work = np.delete(gammas_work, _gwi)
                        gammas_work = np.append(gammas_work, _gv)

                elif probe_lam > 0.5 and cf_best_lam <= 0.5:
                    confirm[t] = 1
                    ltm.append(buffer[probe_idx])
                    mode = 'exploit'
                    actor_idx = probe_idx

                    max_buffer_len = int(p.N) + 1  # dummy + mon_size monitored
                    if len(buffer) > max_buffer_len:
                        post_work, actor_idx, lam_prev_aligned = drop_oldest_non_dummy(
                            post_work,
                            actor_idx,
                            w_extra=lam_prev,
                        )
                        if lam_prev_aligned is None:
                            raise RuntimeError('Failed to align previous lambda prior during confirm cleanup.')
                        lam_prev = lam_prev_aligned
                        gammas_work = np.delete(gammas_work, 0)

                    probe_idx = None

            # Renormalize posterior after reject/confirm cleanup edits.
            post_total = post0_work + float(post_work.sum())
            if post_total <= 0.0 or not np.isfinite(post_total):
                K = len(post_work) + 1
                post0_work = 1.0 / K
                post_work = np.full(len(post_work), 1.0 / K, dtype=float)
            else:
                post0_work = float(post0_work / post_total)
                post_work = post_work / post_total

            # CLEF exploitation policy:
            # keep actor if reliable; otherwise switch to reliable alternative or launch probe.
            if mode == 'exploit':
                kexp = max(float(p.kappa_explore), 1e-6)
                kswitch = max(float(p.kappa_switch), 1e-6)
                actor_wi = int(actor_idx - 1)
                if actor_wi < 0 or actor_wi >= len(post_work):
                    actor_wi = int(np.argmax(post_work))
                    actor_idx = actor_wi + 1
                actor_rel = float(post_work[actor_wi])
                best_wi = int(np.argmax(post_work))
                best_rel = float(post_work[best_wi]) if len(post_work) > 0 else 0.0
                best_buf_idx = int(best_wi + 1)
                p_switch_now = (
                    _sigmoid((best_rel - actor_rel) / kswitch)
                    if best_buf_idx != actor_idx
                    else 0.0
                )
                p_switch_trace[t] = float(p_switch_now)
                p_explore_trace[t] = float(_sigmoid((0.5 - actor_rel) / kexp))

                # Keep hard CLEF dynamics for state updates, use smooth gates only for LL/diagnostics.
                if actor_rel < 0.5:
                    if best_buf_idx != actor_idx and best_rel > 0.5:
                        switch_in[t] = 1
                        switch_to_alt[t] = 1
                        post_work, actor_idx, lam_prev_aligned = move_non_dummy_to_end(
                            post_work,
                            best_buf_idx,
                            w_extra=lam_prev,
                        )
                        if lam_prev_aligned is None:
                            raise RuntimeError('Failed to align previous lambda prior during exploit switch.')
                        lam_prev = lam_prev_aligned
                        # Keep gammas_work aligned with post-cleanup buffer.
                        _gwi = best_buf_idx - 1
                        _gv = float(gammas_work[_gwi])
                        gammas_work = np.delete(gammas_work, _gwi)
                        gammas_work = np.append(gammas_work, _gv)
                    else:
                        switch_in[t] = 1
                        switch_to_probe[t] = 1
                        pending_switch_in = True
            else:
                p_switch_trace[t] = 0.0

            # Optional post-cleanup traces: posterior after reject/confirm edits.
            if len(post_work) == 0:
                raise RuntimeError('No monitored non-dummy strategies in posterior.')
            actor_wi_trace = int(np.clip(actor_idx - 1, 0, len(post_work) - 1))
            (
                _,
                lam_actor_post_cleanup[t],
                lam_alt1_post_cleanup[t],
                lam_alt2_post_cleanup[t],
                lambda_uncertainty_post_cleanup[t],
                lambda_competition_post_cleanup[t],
                lambda_top2_gap_post_cleanup[t],
                lambda_entropy_post_cleanup[t],
                lambda_K_post_cleanup[t],
                lambda_entropy_norm_post_cleanup[t],
            ) = lambda_features(post0_work, post_work, actor_wi_trace)

            if lam_prev.shape != post_work.shape:
                raise RuntimeError('Internal lambda shape mismatch after cleanup.')
            if gammas_work.shape != post_work.shape:
                raise RuntimeError('Internal gammas_work shape mismatch after cleanup.')
            lam_prev0, lam_prev = self._full_normalize(lam_prev0, lam_prev)

            # Donoso SM — Absolute reliability update (literal formula):
            #   μ_i(t) = γ_i(o_t, s_t, a_t) · λ_i(t)
            #   λ_i(t+1) ∝ κ(i, s_{t+1}) · [(1-τ)·μ_i(t) + τ·Σ_{j≠i} μ_j(t)]
            # κ is uniform (no context cues in Donoso task) → cancels in normalization.
            n_monitored_now = len(buffer) - 1
            if n_monitored_now <= 0:
                raise RuntimeError('Buffer is empty during hazard transition.')
            n_lam = len(lam_prev)
            if n_lam <= 0:
                raise RuntimeError('Empty belief distribution at hazard transition.')
            # Step 1: Compute unnormalized posteriors μ_i = γ_i · λ_i.
            mu0 = gamma0 * lam_prev0
            mu = gammas_work * lam_prev
            mu_total = mu0 + float(mu.sum())
            # Step 2: Volatility transition — stay (1-τ) or switch to another (τ).
            # τ-leak is evidence-weighted (Σ_{j≠i} μ_j), NOT uniform.
            lam0 = (1.0 - p.tau) * mu0 + p.tau * (mu_total - mu0)
            lam = (1.0 - p.tau) * mu + p.tau * (mu_total - mu)
            # Step 3: Normalize to probability distribution.
            lam0, lam = self._full_normalize(lam0, lam)
            n_monitored[t] = int(len(buffer) - 1)
            n_ltm[t] = int(len(ltm))

        return {
            'll': ll,
            'lam0_pre': lam0_pre_trace,
            'lam0': lam0_trace,
            'lam_actor_pre': lam_actor_pre,
            'lam_actor': lam_actor,
            'lam_alt1_pre': lam_alt1_pre,
            'lam_alt1': lam_alt1,
            'lam_alt2_pre': lam_alt2_pre,
            'lam_alt2': lam_alt2,
            'lambda_uncertainty_pre': lambda_uncertainty_pre,
            'lambda_uncertainty': lambda_uncertainty,
            'lambda_competition_pre': lambda_competition_pre,
            'lambda_competition': lambda_competition,
            'lambda_top2_gap_pre': lambda_top2_gap_pre,
            'lambda_top2_gap': lambda_top2_gap,
            'lambda_entropy_pre': lambda_entropy_pre,
            'lambda_entropy': lambda_entropy,
            'lambda_K_pre': lambda_K_pre,
            'lambda_K': lambda_K,
            'lambda_entropy_norm_pre': lambda_entropy_norm_pre,
            'lambda_entropy_norm': lambda_entropy_norm,
            'lam_actor_post_cleanup': lam_actor_post_cleanup,
            'lam_alt1_post_cleanup': lam_alt1_post_cleanup,
            'lam_alt2_post_cleanup': lam_alt2_post_cleanup,
            'lambda_uncertainty_post_cleanup': lambda_uncertainty_post_cleanup,
            'lambda_competition_post_cleanup': lambda_competition_post_cleanup,
            'lambda_top2_gap_post_cleanup': lambda_top2_gap_post_cleanup,
            'lambda_entropy_post_cleanup': lambda_entropy_post_cleanup,
            'lambda_K_post_cleanup': lambda_K_post_cleanup,
            'lambda_entropy_norm_post_cleanup': lambda_entropy_norm_post_cleanup,
            'q_actor': q_actor,
            'gamma_actor': gamma_actor,
            'rpe_actor': rpe_actor,
            'p_explore': p_explore_trace,
            'p_switch': p_switch_trace,
            'probe_active': probe_active,
            'exploration': exploration,
            'switch_in': switch_in,
            'switch_to_alt': switch_to_alt,
            'switch_to_probe': switch_to_probe,
            'rejection': reject,
            'confirmation': confirm,
            'n_monitored': n_monitored,
            'n_ltm': n_ltm,
        }


# RL model (comparison)

@dataclass
class RLParams:
    alpha: float
    beta: float
    eps: float


def run_rl(
    s: np.ndarray,
    a: np.ndarray,
    o: np.ndarray,
    params: RLParams,
    n_states: int,
    n_actions: int,
    positive_outcome_idx: Optional[int] = None,
) -> float:
    Q = np.zeros((n_states, n_actions), dtype=float)
    ll = 0.0
    if positive_outcome_idx is None:
        reward_idx = int(np.max(o)) if len(o) > 0 else 0
    else:
        reward_idx = int(positive_outcome_idx)
    if reward_idx < 0:
        raise ValueError(f'positive_outcome_idx must be >= 0, got {reward_idx}.')
    for st, at, ot in zip(s, a, o):
        probs = softmax_probs(Q[st], params.beta)
        probs = apply_lapse(probs, params.eps)
        probs = probs / probs.sum()
        ll += np.log(clip_prob(probs[at]))
        r = 1.0 if ot == reward_idx else 0.0
        Q[st, at] = params.alpha * r + (1.0 - params.alpha) * Q[st, at]
    return ll


def fit_ols_regression(
    trajectories: List[Dict],
    param_names: List[str],
    outcome_col: str = 'confirmation',
    aggregate_fn=None,
    test_size: float = 0.2,
) -> Dict:
    """
    ⚠️  EXPERIMENTAL: OLS regression of AGGREGATED model parameters on outcomes.

    Donoso et al. (2014) Table S1 used block-level aggregates:
    - Predictors: mean(λ_actor), mean(switch_entropy), etc. per subject/session
    - Outcomes: proportion confirmed, proportion switched, etc. per subject/session

    Args:
        trajectories: List of trajectory dicts from run_probe (one per subject/session).
                     Each trajectory dict should contain:
                     - param_names: keys mapping to arrays (trial-level) or scalars (block-level)
                     - outcome_col: key with array or scalar outcome

        param_names: List of parameter/regressor names to aggregate. Default uses per-trial
                    traces like 'lam_actor', 'lambda_competition', 'switch_entropy'.

        outcome_col: Column to predict (e.g., 'confirmation', 'switch_in', 'rejection').
                    Should map to per-trial binary indicator. Will be aggregated as mean.

        aggregate_fn: Function to aggregate per-trial traces. Default: np.mean.
                     Example: lambda x: np.nanmean(x) for traces with NaN.

        test_size: Fraction for train/test split. Default: 0.2.

    Returns:
        Dict with:
        - 'betas': np.ndarray, OLS coefficients for each param
        - 'intercept': float
        - 'r_squared': float, R² on test set
        - 'p_values': np.ndarray, p-values for each coeff
        - 'n_blocks': int, number of blocks (subjects/sessions)
        - 'note': str with methodological note

    ⚠️ WARNING: This is for exploratory validation only. Requires:
       1. trajectories is dict-of-dicts (trajectory per subject, not trial-level)
       2. param_names map to per-trial ARRAYS (will be aggregated)
       3. outcome_col maps to per-trial ARRAY (will be aggregated)

    Example (correct usage):
        out_subj1 = run_probe(...)  # returns dict with 'lam_actor': array(100,), etc.
        out_subj2 = run_probe(...)
        ols = fit_ols_regression(
            [out_subj1, out_subj2],
            param_names=['lam_actor', 'lambda_competition'],
            outcome_col='confirmation',  # must be per-trial indicator
            aggregate_fn=np.mean
        )

    Example (incorrect usage — will crash):
        ols = fit_ols_regression(
            [out_subj1, out_subj2],
            param_names=['scanned_blocks'],  # scalar, not array
            outcome_col='switch_in'  # Dimension mismatch: 1 scalar vs 100 trials
        )  # ❌ AssertionError: Data shape mismatch

    References:
        Donoso, M., Collins, A.G., Koechlin, E. (2014). Science 344(6191).
        Supplementary Table S1: OLS of λ on behavioral outcomes.
    """
    if aggregate_fn is None:
        aggregate_fn = np.mean

    try:
        from scipy.stats import t as t_dist
    except ImportError:
        raise ImportError('scipy required for OLS regression.')

    # === STEP 1: Aggregate each trajectory's parameters ===
    X_data = []   # List of param vectors, one per trajectory
    y_data = []   # List of outcome aggregates, one per trajectory
    n_trials_per_traj = []

    for i, traj in enumerate(trajectories):
        if outcome_col not in traj:
            raise ValueError(f'Trajectory {i} missing outcome column: {outcome_col}')

        # Extract and aggregate outcome
        outcome_vals = np.atleast_1d(traj[outcome_col])
        outcome_agg = float(aggregate_fn(outcome_vals))
        n_trials = len(np.atleast_1d(outcome_vals))

        # Extract and aggregate parameters
        param_row = []
        param_lens = []  # Track lengths to detect dimension mismatches
        for pname in param_names:
            if pname not in traj:
                raise ValueError(f'Trajectory {i} missing parameter: {pname}')
            param_vals = np.atleast_1d(traj[pname])
            param_agg = float(aggregate_fn(param_vals))
            param_row.append(param_agg)
            param_lens.append(len(param_vals))

        # Sanity check: all params should have same length (or be scalars)
        # If outcome has N trials, params should have N trials (when aggregated)
        if len(set(param_lens)) > 1 or (param_lens[0] != n_trials and param_lens[0] > 1):
            raise ValueError(
                f'Trajectory {i}: parameter lengths {param_lens} mismatch outcome length {n_trials}. '
                f'Ensure all param_names map to per-trial arrays of same length as outcome_col. '
                f'Example: run_probe returns lam_actor[100,], confirmation[100,] for 100 trials.'
            )

        X_data.append(param_row)
        y_data.append(outcome_agg)
        n_trials_per_traj.append(n_trials)

    X_all = np.array(X_data, dtype=float)  # (n_trajectories, n_params)
    y_all = np.array(y_data, dtype=float)  # (n_trajectories,)

    if len(X_all) < len(param_names) + 2:
        raise ValueError(
            f'Too few trajectories ({len(X_all)}) for OLS with {len(param_names)} predictors. '
            f'Need at least {len(param_names) + 2} blocks. Found {n_trials_per_traj} trials per block.'
        )

    # === STEP 2: Run OLS ===
    # Add intercept
    X_all = np.column_stack([np.ones(len(X_all)), X_all])

    # Train/test split
    n_total = len(X_all)
    n_test = max(1, int(np.round(n_total * test_size)))
    n_train = n_total - n_test
    idx = np.random.permutation(n_total)
    idx_train, idx_test = idx[:n_train], idx[n_train:]

    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_train, y_test = y_all[idx_train], y_all[idx_test]

    # OLS: β = (X'X)^-1 X'y
    try:
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        betas = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Singular matrix; use pseudo-inverse
        betas = np.linalg.pinv(X_train) @ y_train

    # Compute R-squared on test set
    y_pred_test = X_test @ betas
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute standard errors and p-values
    y_pred_train = X_train @ betas
    residuals = y_train - y_pred_train
    mse = np.sum(residuals ** 2) / max(1, n_train - len(param_names) - 1)
    try:
        var_betas = mse * np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    except np.linalg.LinAlgError:
        var_betas = mse * np.linalg.pinv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    se_betas = np.sqrt(np.diag(var_betas))

    t_vals = betas / (se_betas + 1e-10)
    p_vals = 2.0 * (1.0 - t_dist.cdf(np.abs(t_vals), max(1, n_train - len(param_names) - 1)))

    return {
        'betas': betas[1:],  # Exclude intercept
        'intercept': float(betas[0]),
        'r_squared': float(r_squared),
        'p_values': p_vals[1:],  # Exclude intercept
        'n_blocks': int(len(X_data)),
        'n_train': int(n_train),
        'n_test': int(n_test),
        'n_trials_per_block': n_trials_per_traj,
        'param_names': list(param_names),
        'outcome_col': str(outcome_col),
        'aggregate_fn': str(aggregate_fn),
        'note': '⚠️ EXPERIMENTAL: Block-level OLS per Donoso et al. (2014) Table S1. '
                'Aggregates per-trial parameters over trajectories. '
                'Use for exploratory parameter-behavior correlation checks only.',
    }


# Fit helpers

def _probe_bounds() -> List[Tuple[float, float]]:
    return [
        (0.0, 1.0),           # tau
        (0.0, 1.0),           # eta
        (0.0, 1.0),           # theta
        (1e-5, 1.0 - 1e-5),   # alpha_q
        (0.1, 200.0),         # beta
        (0.0, 0.2),           # eps
    ]


def _probe_params_from_x(x: np.ndarray, N: int) -> ProbeParams:
    return ProbeParams(
        N=int(N),
        tau=float(x[0]),
        eta=float(x[1]),
        theta=float(x[2]),
        alpha_q=float(x[3]),
        beta=float(x[4]),
        eps=float(x[5]),
    )


def _log_prior_uniform_in_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> float:
    for v, (lo, hi) in zip(x, bounds):
        if not (lo <= v <= hi):
            return -np.inf
    return 0.0


def _reflect_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        if lo >= hi:
            raise ValueError(f'Invalid bounds for index {i}: ({lo}, {hi})')
        # reflect repeatedly until in-range, then clip for numeric safety
        while y[i] < lo or y[i] > hi:
            if y[i] < lo:
                y[i] = lo + (lo - y[i])
            if y[i] > hi:
                y[i] = hi - (y[i] - hi)
        y[i] = float(np.clip(y[i], lo, hi))
    return y


def _in_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> bool:
    return all(lo <= float(v) <= hi for v, (lo, hi) in zip(x, bounds))


def _logit(u: float, eps: float = 1e-9) -> float:
    uu = float(np.clip(u, eps, 1.0 - eps))
    return float(np.log(uu / (1.0 - uu)))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = float(np.exp(-z))
        return 1.0 / (1.0 + ez)
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def ll_weights(lls: np.ndarray) -> np.ndarray:
    lls = np.asarray(lls, dtype=float)
    finite = np.isfinite(lls)
    if lls.ndim != 1 or lls.size == 0:
        raise ValueError('lls must be a non-empty 1D array.')
    if not np.any(finite):
        raise ValueError('lls has no finite values.')
    m = float(np.max(lls[finite]))
    w = np.zeros_like(lls, dtype=float)
    w[finite] = np.exp(lls[finite] - m)
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError('Failed to compute normalized LL weights.')
    return w / s


def weighted_mean_params(xs: np.ndarray, w: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    w = np.asarray(w, dtype=float)
    if xs.ndim != 2:
        raise ValueError('xs must be a 2D array [n_samples, n_params].')
    if w.ndim != 1 or w.size != xs.shape[0]:
        raise ValueError('w must be a 1D array with len(w)==len(xs).')
    s = float(np.sum(w))
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError('Weights must sum to a positive finite value.')
    w = w / s
    return np.sum(w[:, None] * xs, axis=0)


def mcmc_probe(
    s: np.ndarray,
    a: np.ndarray,
    o: np.ndarray,
    n_states: int,
    n_actions: int,
    n_outcomes: int,
    N: int,
    x_init: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    n_samples: int = 20_000,
    burn_in: int = 5_000,
    thin: int = 5,
    step_scales: Optional[np.ndarray] = None,
    step_log_beta: float = 0.15,
    step_logit_alpha: float = 0.25,
    positive_outcome_idx: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if n_samples <= 0:
        raise ValueError('n_samples must be > 0.')
    if burn_in < 0:
        raise ValueError('burn_in must be >= 0.')
    if thin <= 0:
        raise ValueError('thin must be > 0.')

    if bounds is None:
        bounds = _probe_bounds()
    if len(bounds) != 6:
        raise ValueError(f'Expected 6 bounds for PROBE params, got {len(bounds)}.')

    rng = np.random.default_rng(seed)
    engine = ProbeEngine(n_states, n_actions, n_outcomes)

    x = np.asarray(x_init, dtype=float)
    if x.size != 6:
        raise ValueError(f'x_init must have length 6, got {x.size}.')
    x = _reflect_to_bounds(x, bounds)

    if step_scales is None:
        # tau, eta, theta, alpha_q, beta, eps
        # alpha_q and beta are updated in transformed spaces below.
        step_scales = np.array([0.02, 0.05, 0.05, 0.0, 0.0, 0.01], dtype=float)
    else:
        step_scales = np.asarray(step_scales, dtype=float)
    if step_scales.size != 6:
        raise ValueError(f'step_scales must have length 6, got {step_scales.size}.')
    if np.any(step_scales < 0.0):
        raise ValueError('All step_scales must be >= 0.')
    # Avoid confusion: these dimensions are handled by transformed proposals.
    step_scales = step_scales.copy()
    step_scales[3] = 0.0  # alpha_q
    step_scales[4] = 0.0  # beta
    if step_log_beta <= 0.0:
        raise ValueError('step_log_beta must be > 0.')
    if step_logit_alpha <= 0.0:
        raise ValueError('step_logit_alpha must be > 0.')

    def log_post(xvec: np.ndarray) -> float:
        lp = _log_prior_uniform_in_bounds(xvec, bounds)
        if not np.isfinite(lp):
            return -np.inf
        p_now = _probe_params_from_x(xvec, int(N))
        out_now = engine.run(s, a, o, p_now, positive_outcome_idx=positive_outcome_idx)
        return float(out_now['ll']) + lp

    cur_lp = float(log_post(x))
    accepts = 0

    xs: List[np.ndarray] = []
    lls: List[float] = []

    total_iters = int(burn_in + n_samples * thin)
    for it in range(total_iters):
        prop = x + rng.normal(0.0, step_scales, size=x.size)

        # beta: random-walk in log-space
        beta_idx = 4
        logb = float(np.log(max(x[beta_idx], 1e-12)))
        logb_prop = logb + float(rng.normal(0.0, step_log_beta))
        prop[beta_idx] = float(np.exp(logb_prop))

        # alpha_q: random-walk in logit-space
        alpha_idx = 3
        za = _logit(float(x[alpha_idx]))
        za_prop = za + float(rng.normal(0.0, step_logit_alpha))
        prop[alpha_idx] = _sigmoid(za_prop)

        # Keep uniform-in-bounds prior exact: out-of-bounds proposals are rejects.
        if not _in_bounds(prop, bounds):
            prop_lp = -np.inf
        else:
            prop_lp = float(log_post(prop))
            if not np.isfinite(prop_lp):
                prop_lp = -np.inf

        if np.isfinite(prop_lp):
            # Hastings term: log q(cur | prop) - log q(prop | cur).
            # For log(beta)-RW this equals log(beta_prop) - log(beta_cur).
            log_q_cur_given_prop_minus_prop_given_cur = (
                np.log(prop[beta_idx]) - np.log(x[beta_idx])
                + np.log(prop[alpha_idx] * (1.0 - prop[alpha_idx]))
                - np.log(x[alpha_idx] * (1.0 - x[alpha_idx]))
            )

            if np.log(rng.random()) < (
                prop_lp - cur_lp + float(log_q_cur_given_prop_minus_prop_given_cur)
            ):
                x = prop
                cur_lp = prop_lp
                accepts += 1

        if it >= burn_in and ((it - burn_in) % thin == 0):
            xs.append(x.copy())
            lls.append(cur_lp)

    xs_arr = np.asarray(xs, dtype=float)
    lls_arr = np.asarray(lls, dtype=float)
    accept_rate = accepts / float(total_iters)
    return xs_arr, lls_arr, float(accept_rate)


def weighted_average_outputs(
    s: np.ndarray,
    a: np.ndarray,
    o: np.ndarray,
    n_states: int,
    n_actions: int,
    n_outcomes: int,
    N: int,
    xs: np.ndarray,
    lls: np.ndarray,
    n_resamples: int = 300,
    positive_outcome_idx: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray | float]:
    xs = np.asarray(xs, dtype=float)
    lls = np.asarray(lls, dtype=float)
    if xs.ndim != 2 or xs.shape[0] == 0:
        raise ValueError('xs must be a non-empty 2D array.')
    if lls.ndim != 1 or lls.size != xs.shape[0]:
        raise ValueError('lls must be 1D with same length as xs.')
    if n_resamples <= 0:
        raise ValueError('n_resamples must be > 0.')

    rng = np.random.default_rng(seed)
    engine = ProbeEngine(n_states, n_actions, n_outcomes)
    w_all = ll_weights(lls)

    n_pick = min(int(n_resamples), int(xs.shape[0]))
    idx = rng.choice(xs.shape[0], size=n_pick, replace=True, p=w_all)

    out_acc: Optional[Dict[str, np.ndarray | float]] = None
    w_sum = 0.0
    for j in idx:
        p_now = _probe_params_from_x(xs[j], int(N))
        out_now = engine.run(s, a, o, p_now, positive_outcome_idx=positive_outcome_idx)
        wj = float(w_all[j])

        if out_acc is None:
            out_acc = {
                k: (np.zeros_like(v, dtype=float) if isinstance(v, np.ndarray) else 0.0)
                for k, v in out_now.items()
            }

        for k, v in out_now.items():
            if isinstance(v, np.ndarray):
                out_acc[k] += wj * np.asarray(v, dtype=float)
            else:
                out_acc[k] += wj * float(v)

        w_sum += wj

    if out_acc is None or w_sum <= 0.0:
        raise RuntimeError('Failed to accumulate weighted outputs.')

    inv = 1.0 / float(w_sum)
    for k in list(out_acc.keys()):
        out_acc[k] = out_acc[k] * inv  # type: ignore[operator]

    return out_acc


def _weighted_param_quantiles(
    xs: np.ndarray,
    w: np.ndarray,
    probs: Tuple[float, ...] = (0.05, 0.5, 0.95),
) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    w = np.asarray(w, dtype=float)
    if xs.ndim != 2 or xs.shape[0] == 0:
        raise ValueError('xs must be a non-empty 2D array.')
    if w.ndim != 1 or w.size != xs.shape[0]:
        raise ValueError('w must be a 1D array with len(w)==len(xs).')
    if np.any(w < 0.0):
        raise ValueError('w must be non-negative.')
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError('w must sum to a positive value.')
    w = w / w_sum

    q = np.asarray(probs, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError(f'Quantile probs must be in [0, 1], got {probs}.')

    out = np.zeros((q.size, xs.shape[1]), dtype=float)
    for j in range(xs.shape[1]):
        order = np.argsort(xs[:, j])
        xj = xs[order, j]
        wj = w[order]
        cdf = np.cumsum(wj)
        cdf[-1] = 1.0
        out[:, j] = np.interp(q, cdf, xj)
    return out


def fit_probe_mcmc(
    s: np.ndarray,
    a: np.ndarray,
    o: np.ndarray,
    n_states: int,
    n_actions: int,
    n_outcomes: int,
    N: int = 3,
    x_init: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    n_samples: int = 20_000,
    burn_in: int = 5_000,
    thin: int = 5,
    step_scales: Optional[np.ndarray] = None,
    step_log_beta: float = 0.15,
    step_logit_alpha: float = 0.25,
    n_resamples: int = 300,
    positive_outcome_idx: Optional[int] = None,
    seed: int = 0,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], Dict[str, object]]:
    if bounds is None:
        bounds = _probe_bounds()
    if x_init is None:
        x_init = np.array([0.06, 0.7, 0.8, 0.4, 40.0, 0.05], dtype=float)

    xs, lls, accept_rate = mcmc_probe(
        s=s,
        a=a,
        o=o,
        n_states=n_states,
        n_actions=n_actions,
        n_outcomes=n_outcomes,
        N=int(N),
        x_init=np.asarray(x_init, dtype=float),
        bounds=bounds,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        step_scales=step_scales,
        step_log_beta=step_log_beta,
        step_logit_alpha=step_logit_alpha,
        positive_outcome_idx=positive_outcome_idx,
        seed=seed,
    )
    w = ll_weights(lls)
    x_bar = weighted_mean_params(xs, w)
    q_arr = _weighted_param_quantiles(xs, w, probs=(0.05, 0.5, 0.95))
    pnames = ['tau', 'eta', 'theta', 'alpha_q', 'beta', 'eps']
    q_summary = {
        name: {
            'q05': float(q_arr[0, i]),
            'q50': float(q_arr[1, i]),
            'q95': float(q_arr[2, i]),
        }
        for i, name in enumerate(pnames)
    }
    p_bar = _probe_params_from_x(x_bar, int(N))
    out_bar = weighted_average_outputs(
        s=s,
        a=a,
        o=o,
        n_states=n_states,
        n_actions=n_actions,
        n_outcomes=n_outcomes,
        N=int(N),
        xs=xs,
        lls=lls,
        n_resamples=n_resamples,
        positive_outcome_idx=positive_outcome_idx,
        seed=seed + 1,
    )
    meta: Dict[str, object] = {
        'accept_rate': float(accept_rate),
        'samples': xs,
        'lls': lls,
        'weights': w,
        'x_bar': x_bar,
        'x_quantiles': q_summary,
    }
    return p_bar, out_bar, meta


def _resolve_n_jobs(n_restarts: int, n_jobs: Optional[int]) -> int:
    if n_restarts <= 1:
        return 1
    n_cpu = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) == -1:
        return max(1, min(n_restarts, n_cpu))
    j = int(n_jobs)
    if j < -1:
        # joblib convention: -2 means n_cpu-1, -3 means n_cpu-2, etc.
        j = max(1, n_cpu + 1 + j)
    return max(1, min(n_restarts, j))


def _normalize_probe_optimizer(method: str) -> str:
    m = str(method).strip().lower()
    aliases = {
        'powell': 'Powell',
        'lbfgsb': 'L-BFGS-B',
        'l-bfgs-b': 'L-BFGS-B',
    }
    if m not in aliases:
        raise ValueError(
            f'Unsupported PROBE optimizer "{method}". Use one of: {sorted(aliases.keys())}.'
        )
    return aliases[m]


def _fit_probe_restart(task):
    (
        x_start,
        s,
        a,
        o,
        n_states,
        n_actions,
        n_outcomes,
        N,
        bounds,
        positive_outcome_idx,
        optimizer_method,
        optimizer_options,
    ) = task
    engine = ProbeEngine(n_states, n_actions, n_outcomes)

    def neg_ll(x: np.ndarray) -> float:
        p = ProbeParams(
            N=N,
            tau=float(x[0]),
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        out = engine.run(s, a, o, p, positive_outcome_idx=positive_outcome_idx)
        return -float(out['ll'])

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def _fit_probe_shared_tau_restart(task):
    (
        x_start,
        s_s,
        a_s,
        o_s,
        n_states_s,
        n_actions_s,
        n_outcomes_s,
        s_c,
        a_c,
        o_c,
        n_states_c,
        n_actions_c,
        n_outcomes_c,
        N,
        bounds,
        positive_outcome_idx_s,
        positive_outcome_idx_c,
        optimizer_method,
        optimizer_options,
    ) = task

    engine_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s)
    engine_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c)

    def neg_joint_ll(x: np.ndarray) -> float:
        # x = [tau_s, tau_c, eta, theta, alpha_q, beta, eps]
        p_s = ProbeParams(
            N=int(N),
            tau=float(x[0]),
            eta=float(x[2]),
            theta=float(x[3]),
            alpha_q=float(x[4]),
            beta=float(x[5]),
            eps=float(x[6]),
        )
        p_c = ProbeParams(
            N=int(N),
            tau=float(x[1]),
            eta=float(x[2]),
            theta=float(x[3]),
            alpha_q=float(x[4]),
            beta=float(x[5]),
            eps=float(x[6]),
        )
        out_s = engine_s.run(s_s, a_s, o_s, p_s, positive_outcome_idx=positive_outcome_idx_s)
        out_c = engine_c.run(s_c, a_c, o_c, p_c, positive_outcome_idx=positive_outcome_idx_c)
        return -(float(out_s['ll']) + float(out_c['ll']))

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_joint_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def fit_probe_shared_tau(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    tau_s_bounds: Tuple[float, float] = (0.0, 0.3),
    tau_c_bounds: Tuple[float, float] = (0.3, 1.0),
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Joint fit with shared eta/theta/alpha_q/beta/eps and block-specific taus."""
    rng = np.random.default_rng(seed)
    _ = _normalize_probe_optimizer(probe_optimizer)

    base_bounds = _probe_bounds()
    bounds: List[Tuple[float, float]] = [
        (float(tau_s_bounds[0]), float(tau_s_bounds[1])),  # tau_s
        (float(tau_c_bounds[0]), float(tau_c_bounds[1])),  # tau_c
        base_bounds[1],  # eta
        base_bounds[2],  # theta
        base_bounds[3],  # alpha_q
        base_bounds[4],  # beta
        base_bounds[5],  # eps
    ]
    for i, (lo, hi) in enumerate(bounds):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f'Invalid bounds at index {i}: ({lo}, {hi})')

    if x0 is None:
        # [tau_s, tau_c, eta, theta, alpha_q, beta, eps]
        x0 = np.array([0.15, 0.6, 0.3, 0.5, 0.2, 3.0, 0.02], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != 7:
        raise ValueError(f'x0 must have length 7, got {x0.size}.')
    x0 = _reflect_to_bounds(x0, bounds)

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s_s,
            a_s,
            o_s,
            int(n_states_s),
            int(n_actions_s),
            int(n_outcomes_s),
            s_c,
            a_c,
            o_c,
            int(n_states_c),
            int(n_actions_c),
            int(n_outcomes_c),
            int(N),
            bounds,
            positive_outcome_idx_s,
            positive_outcome_idx_c,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]

    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_shared_tau_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_shared_tau_restart, tasks))
        except Exception:
            results = [_fit_probe_shared_tau_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = np.asarray(best_res.x, dtype=float)

    p_hat_s = ProbeParams(
        N=int(N),
        tau=float(best_x[0]),
        eta=float(best_x[2]),
        theta=float(best_x[3]),
        alpha_q=float(best_x[4]),
        beta=float(best_x[5]),
        eps=float(best_x[6]),
    )
    p_hat_c = ProbeParams(
        N=int(N),
        tau=float(best_x[1]),
        eta=float(best_x[2]),
        theta=float(best_x[3]),
        alpha_q=float(best_x[4]),
        beta=float(best_x[5]),
        eps=float(best_x[6]),
    )

    out_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s).run(
        s_s,
        a_s,
        o_s,
        p_hat_s,
        positive_outcome_idx=positive_outcome_idx_s,
    )
    out_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c).run(
        s_c,
        a_c,
        o_c,
        p_hat_c,
        positive_outcome_idx=positive_outcome_idx_c,
    )

    try:
        best_res['shared_tau_fit'] = True
        best_res['x_names'] = ['tau_s', 'tau_c', 'eta', 'theta', 'alpha_q', 'beta', 'eps']
        best_res['x_hat'] = best_x
        best_res['ll_s'] = float(out_s['ll'])
        best_res['ll_c'] = float(out_c['ll'])
        best_res['ll_joint'] = float(out_s['ll']) + float(out_c['ll'])
    except Exception:
        pass

    return p_hat_s, out_s, p_hat_c, out_c, best_res


def _fit_probe_shared_single_tau_restart(task):
    (
        x_start,
        s_s,
        a_s,
        o_s,
        n_states_s,
        n_actions_s,
        n_outcomes_s,
        s_c,
        a_c,
        o_c,
        n_states_c,
        n_actions_c,
        n_outcomes_c,
        N,
        bounds,
        positive_outcome_idx_s,
        positive_outcome_idx_c,
        optimizer_method,
        optimizer_options,
    ) = task

    engine_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s)
    engine_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c)

    def neg_joint_ll(x: np.ndarray) -> float:
        # x = [tau, eta, theta, alpha_q, beta, eps]
        tau = float(x[0])
        p_s = ProbeParams(
            N=int(N),
            tau=tau,
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        p_c = ProbeParams(
            N=int(N),
            tau=tau,
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        out_s = engine_s.run(s_s, a_s, o_s, p_s, positive_outcome_idx=positive_outcome_idx_s)
        out_c = engine_c.run(s_c, a_c, o_c, p_c, positive_outcome_idx=positive_outcome_idx_c)
        return -(float(out_s['ll']) + float(out_c['ll']))

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_joint_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def fit_probe_shared_single_tau(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    tau_bounds: Tuple[float, float] = (0.0, 1.0),
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Joint fit with a single shared tau and shared eta/theta/alpha_q/beta/eps."""
    rng = np.random.default_rng(seed)
    _ = _normalize_probe_optimizer(probe_optimizer)

    base_bounds = _probe_bounds()
    bounds: List[Tuple[float, float]] = [
        (float(tau_bounds[0]), float(tau_bounds[1])),  # shared tau
        base_bounds[1],  # eta
        base_bounds[2],  # theta
        base_bounds[3],  # alpha_q
        base_bounds[4],  # beta
        base_bounds[5],  # eps
    ]
    for i, (lo, hi) in enumerate(bounds):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f'Invalid bounds at index {i}: ({lo}, {hi})')

    if x0 is None:
        # [tau, eta, theta, alpha_q, beta, eps]
        x0 = np.array([0.06, 0.7, 0.8, 0.4, 40.0, 0.05], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != 6:
        raise ValueError(f'x0 must have length 6, got {x0.size}.')
    x0 = _reflect_to_bounds(x0, bounds)

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s_s,
            a_s,
            o_s,
            int(n_states_s),
            int(n_actions_s),
            int(n_outcomes_s),
            s_c,
            a_c,
            o_c,
            int(n_states_c),
            int(n_actions_c),
            int(n_outcomes_c),
            int(N),
            bounds,
            positive_outcome_idx_s,
            positive_outcome_idx_c,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]

    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_shared_single_tau_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_shared_single_tau_restart, tasks))
        except Exception:
            results = [_fit_probe_shared_single_tau_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = np.asarray(best_res.x, dtype=float)
    tau_hat = float(best_x[0])

    p_hat_s = ProbeParams(
        N=int(N),
        tau=tau_hat,
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )
    p_hat_c = ProbeParams(
        N=int(N),
        tau=tau_hat,
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )

    out_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s).run(
        s_s,
        a_s,
        o_s,
        p_hat_s,
        positive_outcome_idx=positive_outcome_idx_s,
    )
    out_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c).run(
        s_c,
        a_c,
        o_c,
        p_hat_c,
        positive_outcome_idx=positive_outcome_idx_c,
    )

    try:
        best_res['shared_single_tau_fit'] = True
        best_res['x_names'] = ['tau', 'eta', 'theta', 'alpha_q', 'beta', 'eps']
        best_res['x_hat'] = best_x
        best_res['tau_hat'] = float(tau_hat)
        best_res['ll_s'] = float(out_s['ll'])
        best_res['ll_c'] = float(out_c['ll'])
        best_res['ll_joint'] = float(out_s['ll']) + float(out_c['ll'])
    except Exception:
        pass

    return p_hat_s, out_s, p_hat_c, out_c, best_res


def fit_probe_confirmatory(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    tau_bounds: Tuple[float, float] = (0.0, 1.0),
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Baseline confirmatory model: shared tau/eta/theta/alpha_q/beta/eps across S and C."""
    p_s, out_s, p_c, out_c, meta = fit_probe_shared_single_tau(
        s_s=s_s,
        a_s=a_s,
        o_s=o_s,
        n_states_s=n_states_s,
        n_actions_s=n_actions_s,
        n_outcomes_s=n_outcomes_s,
        s_c=s_c,
        a_c=a_c,
        o_c=o_c,
        n_states_c=n_states_c,
        n_actions_c=n_actions_c,
        n_outcomes_c=n_outcomes_c,
        N=int(N),
        x0=x0,
        n_restarts=n_restarts,
        seed=seed,
        positive_outcome_idx_s=positive_outcome_idx_s,
        positive_outcome_idx_c=positive_outcome_idx_c,
        tau_bounds=tau_bounds,
        probe_optimizer=probe_optimizer,
        probe_optimizer_options=probe_optimizer_options,
        n_jobs=n_jobs,
    )
    qc_s = probe_qc_metrics(out_s)
    qc_c = probe_qc_metrics(out_c)
    try:
        meta['confirmatory_fit'] = True
        meta['qc_S'] = qc_s
        meta['qc_C'] = qc_c
        meta['qc_degenerate_asymmetry'] = bool(
            int(qc_s.get('degenerate_probe', 0)) != int(qc_c.get('degenerate_probe', 0))
        )
    except Exception:
        pass
    return p_s, out_s, p_c, out_c, meta


def _fit_probe_shared_core_separate_beta_eps_restart(task):
    (
        x_start,
        s_s,
        a_s,
        o_s,
        n_states_s,
        n_actions_s,
        n_outcomes_s,
        s_c,
        a_c,
        o_c,
        n_states_c,
        n_actions_c,
        n_outcomes_c,
        N,
        bounds,
        positive_outcome_idx_s,
        positive_outcome_idx_c,
        optimizer_method,
        optimizer_options,
    ) = task

    engine_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s)
    engine_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c)

    def neg_joint_ll(x: np.ndarray) -> float:
        # x = [tau, eta, theta, alpha_q, beta_s, eps_s, beta_c, eps_c]
        p_s = ProbeParams(
            N=int(N),
            tau=float(x[0]),
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        p_c = ProbeParams(
            N=int(N),
            tau=float(x[0]),
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[6]),
            eps=float(x[7]),
        )
        out_s = engine_s.run(s_s, a_s, o_s, p_s, positive_outcome_idx=positive_outcome_idx_s)
        out_c = engine_c.run(s_c, a_c, o_c, p_c, positive_outcome_idx=positive_outcome_idx_c)
        return -(float(out_s['ll']) + float(out_c['ll']))

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_joint_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def fit_probe_shared_core_separate_beta_eps(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    tau_bounds: Tuple[float, float] = (0.0, 1.0),
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Joint fit with shared tau/eta/theta/alpha_q and block-specific beta/eps."""
    rng = np.random.default_rng(seed)
    _ = _normalize_probe_optimizer(probe_optimizer)

    base_bounds = _probe_bounds()
    bounds: List[Tuple[float, float]] = [
        (float(tau_bounds[0]), float(tau_bounds[1])),  # tau (shared)
        base_bounds[1],  # eta (shared)
        base_bounds[2],  # theta (shared)
        base_bounds[3],  # alpha_q (shared)
        base_bounds[4],  # beta_s
        base_bounds[5],  # eps_s
        base_bounds[4],  # beta_c
        base_bounds[5],  # eps_c
    ]
    for i, (lo, hi) in enumerate(bounds):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f'Invalid bounds at index {i}: ({lo}, {hi})')

    if x0 is None:
        # [tau, eta, theta, alpha_q, beta_s, eps_s, beta_c, eps_c]
        x0 = np.array([0.06, 0.7, 0.8, 0.4, 40.0, 0.05, 40.0, 0.05], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != 8:
        raise ValueError(f'x0 must have length 8, got {x0.size}.')
    x0 = _reflect_to_bounds(x0, bounds)

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s_s,
            a_s,
            o_s,
            int(n_states_s),
            int(n_actions_s),
            int(n_outcomes_s),
            s_c,
            a_c,
            o_c,
            int(n_states_c),
            int(n_actions_c),
            int(n_outcomes_c),
            int(N),
            bounds,
            positive_outcome_idx_s,
            positive_outcome_idx_c,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]

    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_shared_core_separate_beta_eps_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_shared_core_separate_beta_eps_restart, tasks))
        except Exception:
            results = [_fit_probe_shared_core_separate_beta_eps_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = np.asarray(best_res.x, dtype=float)

    p_hat_s = ProbeParams(
        N=int(N),
        tau=float(best_x[0]),
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )
    p_hat_c = ProbeParams(
        N=int(N),
        tau=float(best_x[0]),
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[6]),
        eps=float(best_x[7]),
    )

    out_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s).run(
        s_s,
        a_s,
        o_s,
        p_hat_s,
        positive_outcome_idx=positive_outcome_idx_s,
    )
    out_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c).run(
        s_c,
        a_c,
        o_c,
        p_hat_c,
        positive_outcome_idx=positive_outcome_idx_c,
    )

    qc_s = probe_qc_metrics(out_s)
    qc_c = probe_qc_metrics(out_c)
    try:
        best_res['shared_core_separate_beta_eps_fit'] = True
        best_res['x_names'] = [
            'tau',
            'eta',
            'theta',
            'alpha_q',
            'beta_s',
            'eps_s',
            'beta_c',
            'eps_c',
        ]
        best_res['x_hat'] = best_x
        best_res['ll_s'] = float(out_s['ll'])
        best_res['ll_c'] = float(out_c['ll'])
        best_res['ll_joint'] = float(out_s['ll']) + float(out_c['ll'])
        best_res['qc_S'] = qc_s
        best_res['qc_C'] = qc_c
    except Exception:
        pass

    return p_hat_s, out_s, p_hat_c, out_c, best_res


def _fit_probe_fixed_env_restart(task):
    (
        x_start,
        s_s,
        a_s,
        o_s,
        n_states_s,
        n_actions_s,
        n_outcomes_s,
        s_c,
        a_c,
        o_c,
        n_states_c,
        n_actions_c,
        n_outcomes_c,
        N,
        tau_s,
        tau_c,
        bounds,
        positive_outcome_idx_s,
        positive_outcome_idx_c,
        optimizer_method,
        optimizer_options,
    ) = task

    engine_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s)
    engine_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c)

    def neg_joint_ll(x: np.ndarray) -> float:
        # x = [eta, theta, alpha_q, beta, eps]
        p_s = ProbeParams(
            N=int(N),
            tau=float(tau_s),
            eta=float(x[0]),
            theta=float(x[1]),
            alpha_q=float(x[2]),
            beta=float(x[3]),
            eps=float(x[4]),
        )
        p_c = ProbeParams(
            N=int(N),
            tau=float(tau_c),
            eta=float(x[0]),
            theta=float(x[1]),
            alpha_q=float(x[2]),
            beta=float(x[3]),
            eps=float(x[4]),
        )
        out_s = engine_s.run(s_s, a_s, o_s, p_s, positive_outcome_idx=positive_outcome_idx_s)
        out_c = engine_c.run(s_c, a_c, o_c, p_c, positive_outcome_idx=positive_outcome_idx_c)
        return -(float(out_s['ll']) + float(out_c['ll']))

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_joint_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def fit_probe_fixed_env(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    tau_s_env: float,
    tau_c_env: float,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Joint fit with fixed tau from environment and shared eta/theta/alpha_q/beta/eps."""
    tau_s = float(np.clip(tau_s_env, 0.0, 1.0))
    tau_c = float(np.clip(tau_c_env, 0.0, 1.0))
    if not np.isfinite(tau_s) or not np.isfinite(tau_c):
        raise ValueError(f'Invalid fixed taus: tau_s_env={tau_s_env}, tau_c_env={tau_c_env}.')

    rng = np.random.default_rng(seed)
    _ = _normalize_probe_optimizer(probe_optimizer)
    base_bounds = _probe_bounds()
    bounds: List[Tuple[float, float]] = [
        base_bounds[1],  # eta
        base_bounds[2],  # theta
        base_bounds[3],  # alpha_q
        base_bounds[4],  # beta
        base_bounds[5],  # eps
    ]

    if x0 is None:
        # [eta, theta, alpha_q, beta, eps]
        x0 = np.array([0.3, 0.5, 0.2, 3.0, 0.02], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != 5:
        raise ValueError(f'x0 must have length 5, got {x0.size}.')
    x0 = _reflect_to_bounds(x0, bounds)

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s_s,
            a_s,
            o_s,
            int(n_states_s),
            int(n_actions_s),
            int(n_outcomes_s),
            s_c,
            a_c,
            o_c,
            int(n_states_c),
            int(n_actions_c),
            int(n_outcomes_c),
            int(N),
            tau_s,
            tau_c,
            bounds,
            positive_outcome_idx_s,
            positive_outcome_idx_c,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]

    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_fixed_env_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_fixed_env_restart, tasks))
        except Exception:
            results = [_fit_probe_fixed_env_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = np.asarray(best_res.x, dtype=float)

    p_hat_s = ProbeParams(
        N=int(N),
        tau=tau_s,
        eta=float(best_x[0]),
        theta=float(best_x[1]),
        alpha_q=float(best_x[2]),
        beta=float(best_x[3]),
        eps=float(best_x[4]),
    )
    p_hat_c = ProbeParams(
        N=int(N),
        tau=tau_c,
        eta=float(best_x[0]),
        theta=float(best_x[1]),
        alpha_q=float(best_x[2]),
        beta=float(best_x[3]),
        eps=float(best_x[4]),
    )

    out_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s).run(
        s_s,
        a_s,
        o_s,
        p_hat_s,
        positive_outcome_idx=positive_outcome_idx_s,
    )
    out_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c).run(
        s_c,
        a_c,
        o_c,
        p_hat_c,
        positive_outcome_idx=positive_outcome_idx_c,
    )

    try:
        best_res['fixed_env_tau_fit'] = True
        best_res['x_names'] = ['eta', 'theta', 'alpha_q', 'beta', 'eps']
        best_res['x_hat'] = best_x
        best_res['tau_s_env'] = float(tau_s)
        best_res['tau_c_env'] = float(tau_c)
        best_res['ll_s'] = float(out_s['ll'])
        best_res['ll_c'] = float(out_c['ll'])
        best_res['ll_joint'] = float(out_s['ll']) + float(out_c['ll'])
    except Exception:
        pass

    return p_hat_s, out_s, p_hat_c, out_c, best_res


def _fit_probe_fixed_env_bias_restart(task):
    (
        x_start,
        s_s,
        a_s,
        o_s,
        n_states_s,
        n_actions_s,
        n_outcomes_s,
        s_c,
        a_c,
        o_c,
        n_states_c,
        n_actions_c,
        n_outcomes_c,
        N,
        h_s,
        h_c,
        bounds,
        positive_outcome_idx_s,
        positive_outcome_idx_c,
        optimizer_method,
        optimizer_options,
    ) = task

    engine_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s)
    engine_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c)

    logit_h_s = _logit(float(h_s))
    logit_h_c = _logit(float(h_c))

    def neg_joint_ll(x: np.ndarray) -> float:
        # x = [k, eta, theta, alpha_q, beta, eps]
        k = float(x[0])
        tau_s = _sigmoid(logit_h_s + k)
        tau_c = _sigmoid(logit_h_c + k)
        p_s = ProbeParams(
            N=int(N),
            tau=float(tau_s),
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        p_c = ProbeParams(
            N=int(N),
            tau=float(tau_c),
            eta=float(x[1]),
            theta=float(x[2]),
            alpha_q=float(x[3]),
            beta=float(x[4]),
            eps=float(x[5]),
        )
        out_s = engine_s.run(s_s, a_s, o_s, p_s, positive_outcome_idx=positive_outcome_idx_s)
        out_c = engine_c.run(s_c, a_c, o_c, p_c, positive_outcome_idx=positive_outcome_idx_c)
        return -(float(out_s['ll']) + float(out_c['ll']))

    method_now = _normalize_probe_optimizer(str(optimizer_method))
    if optimizer_options is None:
        if method_now == 'Powell':
            optimizer_options = {'maxiter': 10_000, 'xtol': 1e-4, 'ftol': 1e-4}
        else:
            optimizer_options = {'maxiter': 5_000}

    return minimize(
        neg_joint_ll,
        x0=x_start,
        bounds=bounds,
        method=method_now,
        options=optimizer_options,
    )


def fit_probe_fixed_env_bias(
    s_s: np.ndarray,
    a_s: np.ndarray,
    o_s: np.ndarray,
    n_states_s: int,
    n_actions_s: int,
    n_outcomes_s: int,
    s_c: np.ndarray,
    a_c: np.ndarray,
    o_c: np.ndarray,
    n_states_c: int,
    n_actions_c: int,
    n_outcomes_c: int,
    h_s_env: float,
    h_c_env: float,
    N: int = 3,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    seed: int = 0,
    positive_outcome_idx_s: Optional[int] = None,
    positive_outcome_idx_c: Optional[int] = None,
    k_bounds: Tuple[float, float] = (-4.0, 4.0),
    probe_optimizer: str = 'powell',
    probe_optimizer_options: Optional[Dict[str, float | int]] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], ProbeParams, Dict[str, np.ndarray | float], object]:
    """Joint fit with fixed environment hazards and one shared bias parameter k."""
    # Keep hazards strictly inside (0,1) for logit transform.
    h_s = float(np.clip(h_s_env, 1e-6, 1.0 - 1e-6))
    h_c = float(np.clip(h_c_env, 1e-6, 1.0 - 1e-6))
    if not np.isfinite(h_s) or not np.isfinite(h_c):
        raise ValueError(f'Invalid hazards: h_s_env={h_s_env}, h_c_env={h_c_env}.')

    rng = np.random.default_rng(seed)
    _ = _normalize_probe_optimizer(probe_optimizer)
    base_bounds = _probe_bounds()
    bounds: List[Tuple[float, float]] = [
        (float(k_bounds[0]), float(k_bounds[1])),  # k
        base_bounds[1],  # eta
        base_bounds[2],  # theta
        base_bounds[3],  # alpha_q
        base_bounds[4],  # beta
        base_bounds[5],  # eps
    ]

    if x0 is None:
        # [k, eta, theta, alpha_q, beta, eps]
        x0 = np.array([0.0, 0.3, 0.5, 0.2, 3.0, 0.02], dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if x0.size != 6:
        raise ValueError(f'x0 must have length 6, got {x0.size}.')
    x0 = _reflect_to_bounds(x0, bounds)

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s_s,
            a_s,
            o_s,
            int(n_states_s),
            int(n_actions_s),
            int(n_outcomes_s),
            s_c,
            a_c,
            o_c,
            int(n_states_c),
            int(n_actions_c),
            int(n_outcomes_c),
            int(N),
            h_s,
            h_c,
            bounds,
            positive_outcome_idx_s,
            positive_outcome_idx_c,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]

    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_fixed_env_bias_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_fixed_env_bias_restart, tasks))
        except Exception:
            results = [_fit_probe_fixed_env_bias_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = np.asarray(best_res.x, dtype=float)
    k_hat = float(best_x[0])
    tau_s_hat = float(_sigmoid(_logit(h_s) + k_hat))
    tau_c_hat = float(_sigmoid(_logit(h_c) + k_hat))

    p_hat_s = ProbeParams(
        N=int(N),
        tau=tau_s_hat,
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )
    p_hat_c = ProbeParams(
        N=int(N),
        tau=tau_c_hat,
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )

    out_s = ProbeEngine(n_states_s, n_actions_s, n_outcomes_s).run(
        s_s,
        a_s,
        o_s,
        p_hat_s,
        positive_outcome_idx=positive_outcome_idx_s,
    )
    out_c = ProbeEngine(n_states_c, n_actions_c, n_outcomes_c).run(
        s_c,
        a_c,
        o_c,
        p_hat_c,
        positive_outcome_idx=positive_outcome_idx_c,
    )

    try:
        best_res['fixed_env_bias_fit'] = True
        best_res['x_names'] = ['k', 'eta', 'theta', 'alpha_q', 'beta', 'eps']
        best_res['x_hat'] = best_x
        best_res['k_hat'] = float(k_hat)
        best_res['h_s_env'] = float(h_s)
        best_res['h_c_env'] = float(h_c)
        best_res['tau_s_hat'] = float(tau_s_hat)
        best_res['tau_c_hat'] = float(tau_c_hat)
        best_res['ll_s'] = float(out_s['ll'])
        best_res['ll_c'] = float(out_c['ll'])
        best_res['ll_joint'] = float(out_s['ll']) + float(out_c['ll'])
    except Exception:
        pass

    return p_hat_s, out_s, p_hat_c, out_c, best_res


def _fit_probe_single_N(s: np.ndarray, a: np.ndarray, o: np.ndarray,
                        n_states: int, n_actions: int, n_outcomes: int,
                        N: int = 3, x0: Optional[np.ndarray] = None,
                        n_restarts: int = 8, seed: int = 0,
                        positive_outcome_idx: Optional[int] = None,
                        probe_optimizer: str = 'powell',
                        probe_optimizer_options: Optional[Dict[str, float | int]] = None,
                        n_jobs: Optional[int] = None) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], object]:
    rng = np.random.default_rng(seed)
    engine = ProbeEngine(n_states, n_actions, n_outcomes)
    _ = _normalize_probe_optimizer(probe_optimizer)

    if x0 is None:
        x0 = np.array([0.2, 0.3, 0.5, 0.2, 3.0, 0.02], dtype=float)

    bounds = _probe_bounds()

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (
            x_start,
            s,
            a,
            o,
            n_states,
            n_actions,
            n_outcomes,
            int(N),
            bounds,
            positive_outcome_idx,
            probe_optimizer,
            probe_optimizer_options,
        )
        for x_start in starts
    ]
    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_probe_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_probe_restart, tasks))
        except Exception:
            # Fallback for environments where process spawning is unavailable.
            results = [_fit_probe_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = best_res.x
    p_hat = ProbeParams(
        N=int(N),
        tau=float(best_x[0]),
        eta=float(best_x[1]),
        theta=float(best_x[2]),
        alpha_q=float(best_x[3]),
        beta=float(best_x[4]),
        eps=float(best_x[5]),
    )
    out = engine.run(s, a, o, p_hat, positive_outcome_idx=positive_outcome_idx)
    return p_hat, out, best_res


def _normalize_n_grid(N: int | Iterable[int]) -> List[int]:
    if isinstance(N, (int, np.integer)):
        n_grid = [int(N)]
    else:
        n_grid = sorted({int(v) for v in N})
    if len(n_grid) == 0:
        raise ValueError("N grid is empty.")
    if any(v < 1 for v in n_grid):
        raise ValueError(f"All N values must be >= 1, got: {n_grid}")
    return n_grid


def _fit_rl_restart(task):
    x_start, s, a, o, n_states, n_actions, bounds, positive_outcome_idx = task

    def neg_ll(x: np.ndarray) -> float:
        p = RLParams(alpha=float(x[0]), beta=float(x[1]), eps=float(x[2]))
        return -run_rl(
            s,
            a,
            o,
            p,
            n_states,
            n_actions,
            positive_outcome_idx=positive_outcome_idx,
        )

    return minimize(neg_ll, x0=x_start, bounds=bounds, method='L-BFGS-B')


def fit_probe_full(s: np.ndarray, a: np.ndarray, o: np.ndarray,
                   n_states: int, n_actions: int, n_outcomes: int,
                   N: int | Iterable[int] = 3, x0: Optional[np.ndarray] = None,
                   n_restarts: int = 8, seed: int = 0,
                   positive_outcome_idx: Optional[int] = None,
                   probe_optimizer: str = 'powell',
                   probe_optimizer_options: Optional[Dict[str, float | int]] = None,
                   n_jobs: Optional[int] = None) -> Tuple[ProbeParams, Dict[str, np.ndarray | float], object]:
    n_grid = _normalize_n_grid(N)
    if len(n_grid) == 1:
        return _fit_probe_single_N(
            s=s,
            a=a,
            o=o,
            n_states=n_states,
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            N=n_grid[0],
            x0=x0,
            n_restarts=n_restarts,
            seed=seed,
            positive_outcome_idx=positive_outcome_idx,
            probe_optimizer=probe_optimizer,
            probe_optimizer_options=probe_optimizer_options,
            n_jobs=n_jobs,
        )

    n_obs = max(1, int(len(s)))
    n_free = 6  # tau, eta, theta, alpha_q, beta, eps
    candidates: List[Tuple[float, float, int, ProbeParams, Dict[str, np.ndarray | float], object]] = []
    for i, n_val in enumerate(n_grid):
        p_hat_i, out_i, res_i = _fit_probe_single_N(
            s=s,
            a=a,
            o=o,
            n_states=n_states,
            n_actions=n_actions,
            n_outcomes=n_outcomes,
            N=n_val,
            x0=x0,
            n_restarts=n_restarts,
            seed=int(seed + 1009 * i),
            positive_outcome_idx=positive_outcome_idx,
            probe_optimizer=probe_optimizer,
            probe_optimizer_options=probe_optimizer_options,
            n_jobs=n_jobs,
        )
        ll_i = float(out_i['ll'])
        bic_i = float(np.log(n_obs) * n_free - 2.0 * ll_i)
        candidates.append((bic_i, ll_i, int(n_val), p_hat_i, out_i, res_i))

    bic_best, ll_best, n_best, p_best, out_best, res_best = min(
        candidates, key=lambda x: (x[0], -x[1], x[2])
    )
    summary = [
        {'N': n_val, 'bic': bic_val, 'll': ll_val}
        for bic_val, ll_val, n_val, _, _, _ in candidates
    ]
    try:
        res_best['selected_N'] = int(n_best)
        res_best['selected_bic'] = float(bic_best)
        res_best['selected_ll'] = float(ll_best)
        res_best['n_grid_summary'] = summary
    except Exception:
        pass
    return p_best, out_best, res_best


def fit_rl(s: np.ndarray, a: np.ndarray, o: np.ndarray,
           n_states: int, n_actions: int,
           x0: Optional[np.ndarray] = None,
           n_restarts: int = 8, seed: int = 0,
           positive_outcome_idx: Optional[int] = None,
           n_jobs: Optional[int] = None) -> Tuple[RLParams, float, object]:
    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.array([0.2, 3.0, 0.02], dtype=float)

    bounds = [
        (1e-5, 1.0 - 1e-5),
        (0.1, 200.0),
        (0.0, 0.2),
    ]

    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float))

    tasks = [
        (x_start, s, a, o, n_states, n_actions, bounds, positive_outcome_idx)
        for x_start in starts
    ]
    n_workers = _resolve_n_jobs(n_restarts, n_jobs)
    if n_workers == 1:
        results = [_fit_rl_restart(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_fit_rl_restart, tasks))
        except Exception:
            # Fallback for environments where process spawning is unavailable.
            results = [_fit_rl_restart(task) for task in tasks]

    best_res = min(results, key=lambda res: float(res.fun))
    best_x = best_res.x

    p_hat = RLParams(alpha=float(best_x[0]), beta=float(best_x[1]), eps=float(best_x[2]))
    ll = run_rl(
        s,
        a,
        o,
        p_hat,
        n_states,
        n_actions,
        positive_outcome_idx=positive_outcome_idx,
    )
    return p_hat, ll, best_res



# Data prep and output helpers

def prepare_behavior_arrays(
    df,
    stim_col: str,
    action_col: str,
    outcome_col: str,
    positive_outcome_value: Optional[int] = None,
    require_positive_outcome_value: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, Dict[str, object]]:
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame')

    stim = pd.to_numeric(df[stim_col], errors='coerce').astype('Int64')
    act = pd.to_numeric(df[action_col], errors='coerce').astype('Int64')
    out = pd.to_numeric(df[outcome_col], errors='coerce').astype('Int64')

    if stim.isna().any() or act.isna().any() or out.isna().any():
        raise RuntimeError('Found NaN in stim/action/outcome columns after coercion')

    stim_vals = sorted(stim.unique().tolist())
    act_vals = sorted(act.unique().tolist())
    out_vals = sorted(out.unique().tolist())

    stim_map = {v: i for i, v in enumerate(stim_vals)}
    act_map = {v: i for i, v in enumerate(act_vals)}
    out_map = {v: i for i, v in enumerate(out_vals)}
    positive_outcome_idx = resolve_positive_outcome_idx(
        out_vals,
        positive_outcome_value,
        require_explicit=bool(require_positive_outcome_value),
    )
    positive_outcome_value_resolved = out_vals[positive_outcome_idx]

    s = stim.map(stim_map).to_numpy(dtype=int)
    a = act.map(act_map).to_numpy(dtype=int)
    o = out.map(out_map).to_numpy(dtype=int)

    return s, a, o, len(stim_vals), len(act_vals), len(out_vals), {
        'stim': stim_map,
        'action': act_map,
        'outcome': out_map,
        'positive_outcome_idx': int(positive_outcome_idx),
        'positive_outcome_value': int(positive_outcome_value_resolved),
    }


def attach_probe_regressors(df, out: Dict[str, np.ndarray | float]):
    out_df = df.copy()
    out_df['lambda0_pre'] = out.get('lam0_pre', out['lam0'])
    out_df['lambda0'] = out['lam0']
    out_df['q_actor'] = out['q_actor']
    out_df['gamma_actor'] = out['gamma_actor']
    out_df['lambda_actor_pre'] = out.get('lam_actor_pre', out['lam_actor'])
    out_df['lambda_actor'] = out['lam_actor']
    out_df['lambda_alt1_pre'] = out.get('lam_alt1_pre', out['lam_alt1'])
    out_df['lambda_alt1'] = out['lam_alt1']
    out_df['lambda_alt2_pre'] = out.get('lam_alt2_pre', out['lam_alt2'])
    out_df['lambda_alt2'] = out['lam_alt2']
    out_df['lambda_uncertainty_pre'] = out.get('lambda_uncertainty_pre', out.get('lambda_uncertainty', np.full(len(out_df), np.nan)))
    out_df['lambda_uncertainty'] = out.get('lambda_uncertainty', np.full(len(out_df), np.nan))
    out_df['lambda_competition_pre'] = out.get('lambda_competition_pre', out.get('lambda_competition', np.full(len(out_df), np.nan)))
    out_df['lambda_competition'] = out.get('lambda_competition', np.full(len(out_df), np.nan))
    out_df['lambda_top2_gap_pre'] = out.get('lambda_top2_gap_pre', out.get('lambda_top2_gap', np.full(len(out_df), np.nan)))
    out_df['lambda_top2_gap'] = out.get('lambda_top2_gap', np.full(len(out_df), np.nan))
    out_df['lambda_entropy_pre'] = out.get('lambda_entropy_pre', out.get('lambda_entropy', np.full(len(out_df), np.nan)))
    out_df['lambda_entropy'] = out.get('lambda_entropy', np.full(len(out_df), np.nan))
    out_df['lambda_entropy_norm_pre'] = out.get('lambda_entropy_norm_pre', out.get('lambda_entropy_norm', np.full(len(out_df), np.nan)))
    out_df['lambda_entropy_norm'] = out.get('lambda_entropy_norm', np.full(len(out_df), np.nan))
    out_df['lambda_K_pre'] = out.get('lambda_K_pre', out.get('lambda_K', np.full(len(out_df), np.nan)))
    out_df['lambda_K'] = out.get('lambda_K', np.full(len(out_df), np.nan))
    out_df['lambda_actor_post_cleanup'] = out.get('lam_actor_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_alt1_post_cleanup'] = out.get('lam_alt1_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_alt2_post_cleanup'] = out.get('lam_alt2_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_uncertainty_post_cleanup'] = out.get('lambda_uncertainty_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_competition_post_cleanup'] = out.get('lambda_competition_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_top2_gap_post_cleanup'] = out.get('lambda_top2_gap_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_entropy_post_cleanup'] = out.get('lambda_entropy_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_entropy_norm_post_cleanup'] = out.get('lambda_entropy_norm_post_cleanup', np.full(len(out_df), np.nan))
    out_df['lambda_K_post_cleanup'] = out.get('lambda_K_post_cleanup', np.full(len(out_df), np.nan))
    out_df['rpe_post'] = out['rpe_actor']
    out_df['p_explore'] = out.get('p_explore', np.full(len(out_df), np.nan))
    out_df['p_switch'] = out.get('p_switch', np.full(len(out_df), np.nan))
    out_df['probe_active_post'] = out['probe_active']
    out_df['switch_in'] = out['switch_in']
    out_df['switch_to_alt'] = out.get('switch_to_alt', np.zeros(len(out_df), dtype=int))
    out_df['switch_to_probe'] = out.get('switch_to_probe', np.zeros(len(out_df), dtype=int))
    out_df['reject'] = out['rejection']
    out_df['confirm'] = out['confirmation']
    out_df['exploration'] = out['exploration']
    out_df['n_monitored_strategies'] = out.get('n_monitored', np.full(len(out_df), np.nan))
    out_df['n_ltm_strategies'] = out.get('n_ltm', np.full(len(out_df), np.nan))
    out_df['switch'] = out_df['switch_in']

    # Donoso-like aliases used in downstream GLM code.
    out_df['Q_actor_post'] = out_df['q_actor']
    out_df['q_actor_post'] = out_df['q_actor']
    out_df['gamma_actor_post'] = out_df['gamma_actor']
    out_df['lambda0_post'] = out_df['lambda0']
    out_df['lambda_actor_post'] = out_df['lambda_actor']
    out_df['lambda_alt1_post'] = out_df['lambda_alt1']
    out_df['lambda_alt2_post'] = out_df['lambda_alt2']
    out_df['lambda_uncertainty_post'] = out_df['lambda_uncertainty']
    out_df['lambda_competition_post'] = out_df['lambda_competition']
    out_df['lambda_top2_gap_post'] = out_df['lambda_top2_gap']
    out_df['lambda_entropy_post'] = out_df['lambda_entropy']
    out_df['lambda_entropy_norm_post'] = out_df['lambda_entropy_norm']
    out_df['lambda_K_post'] = out_df['lambda_K']
    out_df['probe_active_pre'] = out_df['probe_active_post']
    return out_df
