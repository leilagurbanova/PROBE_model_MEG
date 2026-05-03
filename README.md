# PROBE_model_MEG

End-to-end MEG analysis pipeline: from raw recordings to group-level model comparison.
Master's thesis project at HSE University (Computational Neuroscience).

## What this does

A full pipeline for analyzing magnetoencephalography (MEG) data using the PROBE model.
The pipeline goes through six sequential stages, each implemented as a self-contained Jupyter notebook:

1. **`1 MEG Preprocessing.ipynb`** - Loading raw MEG recordings, artifact rejection,
   filtering, epoching, and quality control.
2. **`2 Inverse and forward solutions.ipynb`** - Building forward models (head models,
   source spaces, BEM) and computing inverse solutions to reconstruct cortical sources
   from sensor-level signals.
3. **`3 Extract ROI timecourses.ipynb`** - Extracting source-level time series for
   regions of interest based on anatomical parcellation.
4. **`4 Merge and model implementation.ipynb`** - Merging subject data and fitting the
   PROBE model to the extracted ROI time courses.
5. **`5 Group Analysis.ipynb`** - Group-level statistical analysis of fitted parameters.
6. **`6 GROUP_COMPARE_MODELS.ipynb`** - Model comparison across alternative
   formulations on the group level.

Helper code lives in `back_model.py`. The `Paradigm/` folder contains the experimental
paradigm used during data collection.

## Stack

- **Python** - NumPy, SciPy, MNE-Python, pandas, Matplotlib
- **MATLAB** - used for behavioral protocol
- **Jupyter** - analysis notebooks

## Repository structure

```
.
├── 1 MEG Preprocessing.ipynb
├── 2 Inverse and forward solutions.ipynb
├── 3 Extract ROI timecourses.ipynb
├── 4 Merge and model implementation.ipynb
├── 5 Group Analysis.ipynb
├── 6 GROUP_COMPARE_MODELS.ipynb
├── back_model.py
└── Paradigm/
```

## Notes

The notebooks are designed to be run in order - each stage produces intermediate
artifacts consumed by the next. Subject-level data is not included in the repository.

---

**Author:** Leila Gurbanova · HSE University, Centre for Cognitive Neuroscience
