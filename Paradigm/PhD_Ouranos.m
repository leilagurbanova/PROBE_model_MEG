% OURANOS
% This script generates Hyperion, a 1011*9 matrix for one subject.
% Structure of Hyperion:
% Column 1: episodes (1:24)
% Column 2: task-sets (1:3) or (1:24)
% Column 3: stimuli (1:3)
% Column 4: expected response (1:4)
% Column 5: true (0) and trap (1) trials
% Column 6: jitter stimulus->feedback
% Column 7: jitter feedback->stimulus
% Column 8: runs (1:4)
% Column 9: normal (0) and post-pause (1) trials
%
% ============================================================

function []=PhD_Ouranos(subj,session)

debug = 0;

jittervalues=[1 1.4; 0.4 0.8];
% nrun = 
% ntrialpersess =

fprintf(['===================================\r'])
fprintf(['Subject',num2str(subj),' Session',num2str(session),'\r'])

subject=subj; % optimized for 24 subjects

tic

column1=[];
column2=[];
column3=[];
column4=[];
column5=[];
column6=[];
column7=[];


% ============================================================

% COLUMN 1: EPISODES

fprintf(['Column 1: episodes.\r'])

load Inachos
epilen=E((subj-1)*16+1:(subj-1)*16+16);

column1=[ones(48,1); 2*ones(48,1); 3*ones(48,1); 4*ones(36,1);...
    5*ones(42,1); 6*ones(33,1); 7*ones(36,1); 8*ones(48,1)];
for i=1:16
    column1=[column1;(i+8)*ones(epilen(i),1)];
end

% ============================================================

% COLUMN 2: TASK-SETS

fprintf(['Column 2: task sets.\r'])

if session==1
    
    column2=[3*ones(48,1); 1*ones(48,1); 2*ones(48,1); 3*ones(36,1);...
        1*ones(42,1); 3*ones(33,1);1*ones(36,1); 2*ones(48,1);...
        3*ones(epilen(1),1); 2*ones(epilen(2),1); 1*ones(epilen(3),1); 3*ones(epilen(4),1);...
        2*ones(epilen(5),1); 1*ones(epilen(6),1); 3*ones(epilen(7),1); 2*ones(epilen(8),1);...
        1*ones(epilen(9),1); 2*ones(epilen(10),1); 3*ones(epilen(11),1); 2*ones(epilen(12),1);...
        3*ones(epilen(13),1); 1*ones(epilen(14),1); 2*ones(epilen(15),1); 1*ones(epilen(16),1)];
    
elseif session==2
    
    column2=[ones(48,1); 2*ones(48,1); 3*ones(48,1); 4*ones(36,1);...
        5*ones(42,1); 6*ones(33,1);7*ones(36,1); 8*ones(48,1);...
        9*ones(epilen(1),1); 10*ones(epilen(2),1); 11*ones(epilen(3),1); 12*ones(epilen(4),1);...
        13*ones(epilen(5),1); 14*ones(epilen(6),1); 15*ones(epilen(7),1); 16*ones(epilen(8),1);...
        17*ones(epilen(9),1); 18*ones(epilen(10),1); 19*ones(epilen(11),1); 20*ones(epilen(12),1);...
        21*ones(epilen(13),1); 22*ones(epilen(14),1); 23*ones(epilen(15),1); 24*ones(epilen(16),1)];
    
end


% ============================================================

% COLUMN 3: STIMULI

fprintf(['Column 3: stimuli.\r'])

column3=[];

for i=1:24
    
    length=size(find(column1==i),1);
    
    if length==48
        stimu=[repmat(1,1,16),repmat(2,1,16),repmat(3,1,16)];
        column3=[column3 Shuffle(stimu)];
    elseif length==42
        stimu=[repmat(1,1,14),repmat(2,1,14),repmat(3,1,14)];
        column3=[column3 Shuffle(stimu)];
    elseif length==36
        stimu=[repmat(1,1,12),repmat(2,1,12),repmat(3,1,12)];
        column3=[column3 Shuffle(stimu)];
    elseif length==33
        stimu=[repmat(1,1,11),repmat(2,1,11),repmat(3,1,11)];
        column3=[column3 Shuffle(stimu)];
    end
    
end

column3=column3';

%CountStim counts the number of stim 1, 2, 3 for each episode
CountStim=zeros(24,3);
for j=1:24
    for i=1:3
        CountStim(j,i)=size(find(column3(find(column1==j))==i),1);
    end
end


% ============================================================

% COLUMN 4: EXPECTED RESPONSE

fprintf(['Column 4: expected response.\r'])

if session==1
    load S_Metis
    for i=1:975
        j=column2(i);
        ts=z(3*(subject-1)+j,1:4);
        column4(i)=find(ts==column3(i));
    end
elseif session==2
    load C_Metis
    for i=1:975
        j=column2(i);
        ts=z(24*(subject-1)+j,1:4);
        column4(i)=find(ts==column3(i));
    end
end


% ============================================================

% COLUMN 5: TRUE AND TRAP TRIALS

fprintf(['Column 5: true and trap trials.\r'])

MD_Rhea_04;
column5=TrapTrue;


% ============================================================

% COLUMNS 6 AND 7: JITTERS

% Take the Designline given by the script Chronos.
% Signification of values:
% 1 -> 0.4 s (stimulus->feedback), 0.1 s (feedback->stimulus)
% 2 -> 0.4 s, 1.85 s
% 3 -> 0.4 s, 3.6 s
% 4 -> 2.15 s, 0.1 s
% 5 -> 2.15 s, 1.85 s
% 6 -> 2.15 s, 3.6 s
% 7 -> 3.9 s, 0.1 s
% 8 -> 3.9 s, 1.85 s
% 9 -> 3.9 s, 3.6 s

fprintf(['Columns 6 and 7: jitters.\r'])

% jittervalues=[0.4, 0.1;0.4, 1.85;0.4, 3.6;...
%     2.15, 0.1;2.15, 1.85;2.15, 3.6;...
%     3.9, 0.1;3.9, 1.85;3.9, 3.6];

% Designline=PhD_Chronos(subj);

% column6(1:339)=0.8;
% column7(1:339)=0.8;
% for i=1:636
% column6=jittervalues(Designline(i),1);
% column7=jittervalues(Designline(i),2);
% end

column6 = jittervalues(1,1) + (jittervalues(1,2)-jittervalues(1,1)).*rand(size(column1, 1),1);
column7 = jittervalues(2,1) + (jittervalues(2,2)-jittervalues(2,1)).*rand(size(column1, 1),1);

%%%%%%%%%%%%%%


DesignMatrix=[column1 column2 column3 column4' column5 column6 column7];

% ============================================================

% ASSEMBLING THE FINAL DESIGN MATRIX

% Since now the DesignMatrix is a 648*7 matrix.
% It will be transformed in a 675*9 matrix using Erebos.

fprintf(['Columns 8 and 9: runs and post-pause trials.\r'])

PhD_Erebos;

fprintf(['Assembling the whole design.\r'])

Hyperion=FullDesignMatrix;

clear i j k n r ts1 ts2 ts3 x y z increment thetys
clear column1 column2 column3 column4 column5 column6 column7
clear Designline jittervalues TrapTrue
clear Metis subject
clear DesignMatrix FullDesignMatrix

if ~debug
    if session==1
        fprintf(['Saving HyperionS.\r'])
        save(['HyperionS',num2str(subj)],'Hyperion');
    elseif session==2
        fprintf(['Saving HyperionC.\r'])
        save(['HyperionC',num2str(subj)],'Hyperion');
    end
end
toc

end
