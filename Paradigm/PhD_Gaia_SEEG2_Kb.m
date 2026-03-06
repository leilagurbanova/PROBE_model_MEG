% GAIA
%
% Final Mnemosyne structure contains:
%
% Mnemosyne.ts:
% uootask-sets used for the subject
% Mnemosyne.buttons: buttons associated to these task-sets
% Mnemosyne.configs: visual configuration of the experiment
%
% Mnemosyne.matrix:
% Column 1: episodes (1:16)
% Column 2: task-sets (1:3) or (1:16)
% Column 3: stimuli (1, 2 or 3)
% Column 4: expected response (1:4)
% Column 5: true (0) and trap (1) trials
% Column 6: jitter stimulus->feedback
% Column 7: jitter feedback->stimulus
% Column 8: runs (1:4)
% Column 9: normal (0) and post-pause (1) trials
% Column 10: pressed button (1:4).
% Column 11: code of the pressed button (7, 9, 13, 14).
% Column 12: correct (1) or false (0) trial, or no button pressed (-1).
% Column 13: reward (1) or no reward (0), or no button pressed (-1).
% Column 14: response time.
%
% Using this program: change 'subj', integrate 'Metis' and the correct
% 'Hyperion' in the current directory, run.
%
% PCWIN64 instead of PCWIN    
%
% ============================================================
PhD_Gaia_SEEG2(1,1,1);


function Mnemosyne = PhD_Gaia_SEEG2(subj,session,run)

debug = 1;
os    = computer;

% duration of stim and FB in seconds
stimduration = 0.8;
fbduration   = 0.8;
% prestime=1; % MD stuff ??? deprecated
resplimit    = 0.7; % patients have resplimit + stimduration to give a response
 

% colors used in the task
colorz={[255 0 0],[0 255 0],[0 0 255],[0 255 255],[255 255 0],[255 0 255]}; % color order r g b c y k (up to 6)

% Kb response buttons
if strcmp(os, 'MACI64')
    buttonz=[13 14 15 51]; % D=7, F=9, J=13, K=14, L=15, M=51
elseif strcmp(os, 'PCWIN64')
    buttonz=[KbName('1!') KbName('2@') KbName('3#') KbName('4$')]; %jklm ss win32 french Kb; same for win64 Russian
else
    return;
end   

% stimuli
stim=3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Serial port : collect RT

%if strcmp(os, 'PCWIN64'); portSpecRT = 'Com1'; end
% lineTerminator = 10; % Default to ASCII code 10, aka LF, aka NL aka newline/linefeed:
%baudRate       = 115200;
%portSettingsRT = sprintf('BaudRate=%i', baudRate);
%maxReadQuantum = 1;
%sampleFreq = 120;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Serial port : send EEG triggers

ioObj = io64;
status = io64(ioObj);

if status ~= 0
    error('io64 failed');
end

portAddress = hex2dec('E030');
%if strcmp(os, 'PCWIN64'); portSpecEEG = 'Com1'; end
%baudRate       = 9600;
%portSettingsEEG = sprintf('BaudRate=%i', baudRate);
triggerValue = 200; 
io64(ioObj, portAddress, triggerValue);
pause(0.05);
io64(ioObj, portAddress, 0);

% ============================================================
%% A. HIDDEN PROCESSING: INITIAL COMPUTATIONS
% ============================================================

%% A.1 RECOVERS INFORMATION FROM HYPERION

if session==1
    load(['HyperionS',num2str(subj)]);
elseif session==2
    load(['HyperionC',num2str(subj)]);
end

sprintf('mean S>FB is %g', mean(Hyperion(:,6)) + stimduration)
sprintf('mean FB>S is %g', mean(Hyperion(:,7)) + fbduration)
sprintf('SD S>FB is %g', std(Hyperion(:,6)))
sprintf('SD FB>S is %g', std(Hyperion(:,7)))

blockseq =Hyperion(:,1);
taskseq  =Hyperion(:,2);
stimseq  =Hyperion(:,3);
actionseq=Hyperion(:,4);
dummy    =Hyperion(:,9);

% ============================================================
%% A.2 SETS THE STRUCTURE OF THE EXPERIMENT

% sets the mode (stimuli "1,3,5" or "2,4,6")
modx=repmat([1 2;2 1],20,1);
mode=modx(subj,session);

if mode==1
    stimSet=[1 3 5];
elseif mode==2
    stimSet=[2 4 6];
end

sessions=1;

if run==1
    sesslen=[180]; % length of run 1
elseif run==2
    sesslen=[345]; % length of runs 1+2
elseif run==3
    sesslen=[513]; % etc.
elseif run==4
    sesslen=[678];
elseif run==5
    sesslen=[846];
elseif run==6
    sesslen=[1011];
end

trials=length(stimseq);

blocks=[4];
ChangeTime=[];
for i=1:1010
    if Hyperion(i+1,2)~=Hyperion(i,2)
        blocks=[blocks Hyperion(i+1,2)];
        ChangeTime=[ChangeTime i];
    end
end
ChangeTime=[0 ChangeTime];

% ============================================================

StartStim=[];
StartReward=[];
EndStim=[];
EndReward=[];

% first stim: 2s after the beginning of the scan
% first reward: 2s + 1s + first stim->reward jitter after the first stim
h=Hyperion(find(Hyperion(:,8)==run),6);
h=h(1);
StartStim=[2];
StartReward=[3+h];
EndStim=[2+1];
EndReward=[3+h+1];

lengthrun=size(find(Hyperion(:,8)==run),1);
initrun=find(Hyperion(:,8)==run);
initrun=initrun(1); % position of the first value of the run

for i=2:lengthrun
 

    % Onset(i) = onset(i-1) + stim duration + FB duration + Stim jitter (i-1) + FB jitter (i-1)


    StartStim(i)  = StartStim(i-1)   + stimduration + fbduration + Hyperion(initrun+(i-2),6) + Hyperion(initrun+(i-2),7);
    StartReward(i)= StartReward(i-1) + stimduration + fbduration + Hyperion(initrun+(i-2),7) + Hyperion(initrun+(i-1),6);
%     OffsetRewardon(i) = StartReward(i-1) + stimduration + fbduration + Hyperion(initrun+(i-2),7) + Hyperion(initrun+(i-1),6);

    EndStim(i)  = StartStim(i)   + stimduration;
    EndReward(i)= StartReward(i) + fbduration;
%     OffsetRewardoff(i) =
    
end

% ============================================================

%% A.3 FILLS THE MNEMOSYNE STRUCTURE

data=[];

if session==1
    load S_Metis;
    % Gives the position and code of the buttons associated to the task-sets
    X=z(3*(subj-1)+1:3*(subj-1)+3,:);
    TS=[];
    TSbuttonz=[];
    for i=1:3
        TS=[TS;[find(X(i,:)==1) find(X(i,:)==2) find(X(i,:)==3)]];
        TSbuttonz=[TSbuttonz;buttonz(TS(i,1)) buttonz(TS(i,2)) buttonz(TS(i,3))];
    end
elseif session==2
    load C_Metis;
    % Gives the position and code of the buttons associated to the task-sets
    X=z(24*(subj-1)+1:24*(subj-1)+24,:);
    TS=[];
    TSbuttonz=[];
    for i=1:24
        TS=[TS;[find(X(i,:)==1) find(X(i,:)==2) find(X(i,:)==3)]];
        TSbuttonz=[TSbuttonz;buttonz(TS(i,1)) buttonz(TS(i,2)) buttonz(TS(i,3))];
    end
end

% Gives the spatial configuration of the stimuli
Y=zeros(1011,4);
for i=1:1011
    Y(i,Hyperion(i,4))=Hyperion(i,3);
end

data.ts=TS;
data.buttons=TSbuttonz;
data.configs=Y;

% Open port portSpec with portSettings, return handle:
%com1 = IOPort('OpenSerialPort', portSpecRT);
%IOPort('ConfigureSerialPort', com1, portSettingsRT);

%com3 = IOPort('OpenSerialPort', portSpecEEG, portSettingsEEG);
%IOPort('ConfigureSerialPort', com3, portSettingsEEG);
%%ioObj = io64;
%%status = io64(ioObj);

%%if status ~= 0
%%    error('io64 failed');
%%end

%%portAddress = hex2dec('E030');

% ============================================================
%% B. VISUAL PROCESSING: THE SCREEN OPENS
% ============================================================

% DEFINES THE POSITION OF BOXES AND THE TEXT POLICE

try
    %Screen('Preference', 'SkipSyncTests', 0);   
    Screen('Preference','TextRenderer', 0);
    Screen('Preference', 'VBLTimestampingMode', -1);
    Screen('Preference', 'SkipSyncTests', 2);
    [w, rect] = Screen('OpenWindow', 0, 0,[],32,2);
    
    center=[rect(3)/2 rect(4)/2];
    crect=CenterRectOnPoint([0 0 100 100],rect(3)/2,rect(4)/2);
    % UIOP --> DFJK
    %tirerect=CenterRectOnPoint([0 0 150 30],(rect(3)/2),(rect(4)/2)+100);
    tirerect=CenterRectOnPoint([0 0 250 50],(rect(3)/2),(rect(4)/2));
    keyU=CenterRectOnPoint([0 0 80 80],(rect(3)/2)-120,(rect(4)/2));
    keyI=CenterRectOnPoint([0 0 80 80],(rect(3)/2)-40,(rect(4)/2));
    keyO=CenterRectOnPoint([0 0 80 80],(rect(3)/2)+40,(rect(4)/2));
    keyP=CenterRectOnPoint([0 0 80 80],(rect(3)/2)+120,(rect(4)/2));
    
    keyU2=CenterRectOnPoint([0 0 80 80],(rect(3)/2)-120,(rect(4)/2));
    keyI2=CenterRectOnPoint([0 0 80 80],(rect(3)/2)-40,(rect(4)/2));
    keyO2=CenterRectOnPoint([0 0 80 80],(rect(3)/2)+40,(rect(4)/2));
    keyP2=CenterRectOnPoint([0 0 80 80],(rect(3)/2)+120,(rect(4)/2));
    
    if strcmp(os, 'MACI64')
        point1=[center(1)-140 center(2)-25];
        point2=[center(1)-60 center(2)-25];
        point3=[center(1)+30 center(2)-25];
        point4=[center(1)+110 center(2)-25];
    elseif strcmp(os, 'PCWIN64')
        %point1=[center(1)-145 center(2)-40];
        point1=[center(1)-140 center(2)-35];
        point2=[center(1)-60  center(2)-35];
        point3=[center(1)+20  center(2)-35];
        point4=[center(1)+100  center(2)-35];
    end
    Screen('TextFont', w , 'Arial');
    Screen('TextSize', w, 100 );
    
    % SETS THE NUMBER OF SESSIONS
    
    cor=0;
    rew=0;
    
    if run==1
        t=0; % depart time of run 1
    elseif run==2
        t=180; % depart time of run 2
    elseif run==3
        t=345; % etc.
    elseif run==4
        t=513;
    elseif run==5
        t=678;
    elseif run==6
        t=846;
    end
    
    data.matrix=[Hyperion zeros(1011,7)];
    
    for s=1:sessions
        
        % INTRODUCTION
        
        T1=GetSecs;
        if ~debug; HideCursor; end;
        
        %IOPort('Purge', com1);
        %IOPort('Purge', com3);
        
        if s==1
            Screen('FillRect', w,0);
            Screen('TextSize', w, 32);
            Screen('DrawText', w, 'Press Enter...', (rect(3)/2)-150, rect(4)/2, 255);
            Screen('Flip', w);                                    
            %IOPort('Write', com3, uint8(200));    
            io64(ioObj, portAddress,200);
            pause(0.05);
            io64(ioObj, portAddress, 0);
        else
            Screen('FillRect', w, 0)
            Screen('TextSize', w, 32 );
            Screen('DrawText', w, 'Запомните текущую стратегию перед отдыхом...',(rect(3)/2)-150,rect(4)/2,255);
            Screen('Flip', w);
            %IOPort('Write', com3, uint8(200));
            io64(ioObj, portAddress,200);
            pause(0.05);
            io64(ioObj, portAddress, 0);
        end
        
        % INITIATES THE EXPERIMENT WHEN ONE BUTTON IS PRESSED
        
        kdown=0;
        while kdown==0;
            kdown=KbCheck; % checks if a button was pressed
        end
        %IOPort('Write', com3, uint8(201));
        io64(ioObj, portAddress,201);
        pause(0.05);
        io64(ioObj, portAddress, 0);
                
        Screen('FillRect', w, 0)
        Screen('Flip', w);
        %IOPort('Write', com3, uint8(202));
        io64(ioObj, portAddress,202);
        pause(0.05);
        io64(ioObj, portAddress, 0);
        
        WaitSecs(2)
        
        Initia=GetSecs;  
        %IOPort('Write', com3, uint8(203));
        io64(ioObj, portAddress,203);
        pause(0.05);
        io64(ioObj, portAddress, 0);
        
        tt=0;
        tic
        % DRAWS THE BOXES
        
        while t<sesslen(s);
            
            t=t+1;
            tt=tt+1;
            
            Screen(w,'FillPoly',0, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]); % black screen
            Screen('TextSize', w, 100 );
            Screen(w,'Fillrect',0,keyU);
            Screen(w,'Framerect',255,keyU,5); % white boxes, width 5
            Screen(w,'Fillrect',0,keyI);
            Screen(w,'Framerect',255,keyI,5);
            Screen(w,'Fillrect',0,keyO);
            Screen(w,'Framerect',255,keyO,5);
            Screen(w,'Fillrect',0,keyP);
            Screen(w,'Framerect',255,keyP,5);
            Screen('gluDisk', w, [255 255 255], rect(3)/2, (rect(4)/2), 10);
            
            Screen('TextSize', w, 50 );
            Screen('DrawText', w, [num2str(stimSet(stimseq(t)))],point1(1),point1(2),255);
            Screen('DrawText', w, [num2str(stimSet(stimseq(t)))],point2(1),point2(2),255);
            Screen('DrawText', w, [num2str(stimSet(stimseq(t)))],point3(1),point3(2),255);
            Screen('DrawText', w, [num2str(stimSet(stimseq(t)))],point4(1),point4(2),255);
            
            while GetSecs<Initia+StartStim(tt)
            end
            
            Screen('Flip', w); % affichage
            TempoStim=GetSecs-Initia;                        
            %IOPort('Write', com3, uint8(100 + num2str(stimSet(stimseq(t)))));
            io64(ioObj, portAddress,100 + num2str(stimSet(stimseq(t))));
            pause(0.05);
            io64(ioObj, portAddress, 0);
            tstart=GetSecs;
            
            press=0;
            
            % CHECKS IF A BUTTON WAS PRESSED, AND WHICH ONE
            
            while GetSecs<Initia+EndStim(tt) && press==0 % pendant que le stim est affichй et qu'il n'y a pas de rйponse
                [kdown secs code]=KbCheck;               % checks if a button was pressed, when and which one
                
                if kdown==1; % if a button was pressed
                    
                    press=press+1;
                    if press==1;
                        Rsec=secs;
                        RT=secs-tstart;
                        keycode=find(code==1); % number of the pressed button
                        keycode=keycode(1);    % takes only the first button pressed                        
                        code=0;
                        if intersect(keycode,buttonz);
                            code=find(keycode==buttonz); 
                            %IOPort('Write', com3, uint8(keycode));
                            io64(ioObj, portAddress, keycode);
                            pause(0.05);
                            io64(ioObj, portAddress, 0);
                            
                        end
                    end
                else
                    code=0;
                end
            end
            
            
            % STIMULUS DURATION
            
            pressw=0;
            if code~=0
                WaitSecs(stimduration-RT); % if response while stim displayed, then wait for stimduration to be complete before jitter then FB
                pressw=1;                  % variable pressw indicates that the button was pressed during the stimulus or not
            end
            
            % MAINTAINS THE BOXES BETWEEN STIMULUS AND REWARD
            
            Screen(w,'FillPoly',0, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]); % black screen
            Screen(w,'Fillrect',0,keyU);
            Screen(w,'Framerect',255,keyU,5);
            Screen(w,'Fillrect',0,keyI);
            Screen(w,'Framerect',255,keyI,5);
            Screen(w,'Fillrect',0,keyO);
            Screen(w,'Framerect',255,keyO,5);
            Screen(w,'Fillrect',0,keyP);
            Screen(w,'Framerect',255,keyP,5);
            Screen('gluDisk', w, [255 255 255], rect(3)/2, (rect(4)/2), 10);
            
            Screen('Flip', w);
            
            % CHECKS IF A BUTTON WAS PRESSED, AND WHICH ONE
            
            while GetSecs<Initia+EndStim(tt)+resplimit && press==0
                [kdown secs code]=KbCheck; % checks if a button was pressed, when and which one
                
                if kdown==1; % if a button was pressed
                    
                    press=press+1;
                    if press==1;
                        Rsec=secs;
                        RT=secs-tstart;
                        keycode=find(code==1); % number of the pressed button
                        keycode=keycode(1); % takes only the first button pressed
                        code=0;
                        if intersect(keycode,buttonz);
                            code=find(keycode==buttonz);
                            %IOPort('Write', com3, uint8(keycode)); 
                            io64(ioObj, portAddress, keycode);
                            pause(0.05);
                            io64(ioObj, portAddress, 0);
                        end
                    end
                else
                    code=0;
                end
            end
            
            % CALCULATES CORRECT/UNCORRECT AND REWARD/NO REWARD
            
            if code~=0
                if code==actionseq(t)
                    cor=1;
                    if Hyperion(t,5)==0
                        rew=1;
                        fbcode = 190;
                    else
                        rew=0;
                        fbcode = 191;
                    end
                else
                    cor=0;
                    if Hyperion(t,5)==0
                        rew=0;
                        fbcode = 192;
                    else
                        rew=1;
                        fbcode = 193;
                    end
                end                
            else
                cor=-1;
                rew=-1;
                RT=-1;
                keycode=-5;
                fbcode = 204;
                %IOPort('Write', com3, uint8(fbcode)); % at the end og the response window, no response given
                io64(ioObj, portAddress, fbcode);
                pause(0.05);
                io64(ioObj, portAddress, 0);
                fbcode = fbcode +1;
            end
            
            % STIMULUS->REWARD JITTER
            
%             if code~=0 % button pressed
%                 if pressw==0 % button pressed after the stimulus
%                     WaitSecs(Hyperion(t,6)-(RT-prestime));
%                 elseif pressw==1 % button pressed during the stimulus
%                     WaitSecs(Hyperion(t,6));
%                 end
%             else % button non pressed
%                 WaitSecs(Hyperion(t,6)-0.35);
%             end
            
            % BIP INDICATING THE REWARD
            
%             if rew==1
%                 note2([500 600 700 800],[0.05 0.05 0.05 0.05]); % frequence and length
%             elseif rew==0 | rew==-1
%                 note2([800 700 600 500],[0.05 0.05 0.05 0.05]);
%             end
            
            % GIVES THE REWARD
            
            Screen(w,'FillPoly',0, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]); % black screen
            Screen(w,'Fillrect',0,keyU);
            Screen(w,'Framerect',255,keyU,5);
            Screen(w,'Fillrect',0,keyI);
            Screen(w,'Framerect',255,keyI,5);
            Screen(w,'Fillrect',0,keyO);
            Screen(w,'Framerect',255,keyO,5);
            Screen(w,'Fillrect',0,keyP);
            Screen(w,'Framerect',255,keyP,5);
            Screen('TextSize', w, 50 );
            Screen('gluDisk', w, [255 255 255], rect(3)/2, (rect(4)/2), 10);
            
            if rew==1 % if the trial is won, places the stimulus in the chosen box
                if keycode==buttonz(1)
                    Screen(w,'DrawText',[num2str(stimSet(stimseq(t)))],point1(1),point1(2),[0 255 0],255);
                elseif keycode==buttonz(2)
                    Screen(w,'DrawText',[num2str(stimSet(stimseq(t)))],point2(1),point2(2),[0 255 0],255);
                elseif keycode==buttonz(3)
                    Screen(w,'DrawText',[num2str(stimSet(stimseq(t)))],point3(1),point3(2),[0 255 0],255);
                elseif keycode==buttonz(4)
                    Screen(w,'DrawText',[num2str(stimSet(stimseq(t)))],point4(1),point4(2),[0 255 0],255);
                end
            end
            
            if rew==0 % if the trial is lost, places a cross in the chosen box
                if keycode==buttonz(1)
                    Screen(w,'DrawText',['X'],point1(1),point1(2),[255 0 0],255);
                elseif keycode==buttonz(2)
                    Screen(w,'DrawText',['X'],point2(1),point2(2),[255 0 0],255);
                elseif keycode==buttonz(3)
                    Screen(w,'DrawText',['X'],point3(1),point3(2),[255 0 0],255);
                elseif keycode==buttonz(4)
                    Screen(w,'DrawText',['X'],point4(1),point4(2),[255 0 0],255);
                else
                    Screen(w,'DrawText',['-'],point1(1),point1(2),[255 0 0],255);
                    Screen(w,'DrawText',['-'],point2(1),point2(2),[255 0 0],255);
                    Screen(w,'DrawText',['-'],point3(1),point3(2),[255 0 0],255);
                    Screen(w,'DrawText',['-'],point4(1),point4(2),[255 0 0],255);
                end
            end
            
            if rew==-1 % if there is no answer, places a '-' in all boxes
                Screen(w,'DrawText',['-'],point1(1),point1(2),[255 0 0],255);
                Screen(w,'DrawText',['-'],point2(1),point2(2),[255 0 0],255);
                Screen(w,'DrawText',['-'],point3(1),point3(2),[255 0 0],255);
                Screen(w,'DrawText',['-'],point4(1),point4(2),[255 0 0],255);
            end
            
            while GetSecs<Initia+StartReward(tt)
            end
            
            Screen('Flip', w);
            TempoReward=GetSecs-Initia;
            %IOPort('Write', com3, uint8(fbcode));
            io64(ioObj, portAddress, fbcode);
            pause(0.05);
            io64(ioObj, portAddress, 0);


            % REWARD DURATION
            
            while GetSecs<Initia+EndReward(tt)
            end
            
            % MAINTAINS THE BOXES BETWEEN REWARD AND NEXT STIMULUS
            
            Screen(w,'FillPoly',0, [0 0;0 rect(4);rect(3) rect(4);rect(3) 0]); % ecran total couleur contexte
            Screen(w,'Fillrect',0,keyU);
            Screen(w,'Framerect',255,keyU,5);
            Screen(w,'Fillrect',0,keyI);
            Screen(w,'Framerect',255,keyI,5);
            Screen(w,'Fillrect',0,keyO);
            Screen(w,'Framerect',255,keyO,5);
            Screen(w,'Fillrect',0,keyP);
            Screen(w,'Framerect',255,keyP,5);
            Screen('gluDisk', w, [255 255 255], rect(3)/2, (rect(4)/2), 10);
            Screen('Flip', w);            

            %IOPort('Write', com3, uint8(206));  % turn off FB
            io64(ioObj, portAddress, 206);
            pause(0.05);
            io64(ioObj, portAddress, 0);
            
            
            % REWARD->STIMULUS JITTER
            
            TempoTrial=GetSecs-Initia;
            
            % SAVES THE DATA
            
            data.matrix(t,10)=code;
            data.matrix(t,11)=keycode;
            data.matrix(t,12)=cor;
            data.matrix(t,13)=rew;
            data.matrix(t,14)=RT;
            data.matrix(t,15)=TempoStim;
            data.matrix(t,16)=TempoReward;
            data.matrix(t,17)=TempoTrial;
            
        end
        data.initia=Initia;
        Termine=GetSecs;
        data.termine=Termine;
        %save Mnemosyne data;
        toc
        if session==1
            save(['Mnemosyne_S',num2str(subj),'_run',num2str(run)], 'data');
        elseif session==2
            save(['Mnemosyne_C',num2str(subj),'_run',num2str(run)], 'data');
        end
        
    end
    
    %% EXIT
    
    io64(ioObj, portAddress, 0);
    %IOPort('Purge', com1);
    %IOPort('Purge', com3);
    %IOPort('CloseAll');
    
    ShowCursor;
    Screen('CloseAll');
    
catch
    
    if ~debug;
        if session==1
            save(['Mnemosyne_S_failsafe',num2str(subj),'_run',num2str(run)], 'data');
        elseif session==2
            save(['Mnemosyne_C_failsafe',num2str(subj),'_run',num2str(run)], 'data');
        end
    end
        
    %IOPort('Purge', com1);
    %IOPort('Purge', com3);
    %IOPort('CloseAll');
    io64(ioObj, portAddress, 0);
    
    ShowCursor
    Screen('CloseAll');
    
    rethrow(lasterror);
end

end