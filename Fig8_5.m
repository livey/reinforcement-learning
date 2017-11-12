function main() 
% for test purpose
clear all;close all;
dbstop iff error 

init();


plot8_5();
    

% global actions up right down left
% global height width 
% 
% for ii=1:height
%     for jj = 1:width
%         [r,n]=env([ii,jj],down);
%         fprintf('current state(%d,%d), nextState (%d,%d),r =%d\n',...
%             ii,jj,n(1),n(2),r);
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function init()
% init some global vairables 
% action variable 
global actions right left up down 
right = 1;
left = 2;
up = 3;
down = 4;
actions = [right,left,up,down];

% start and goal 
global startState goalState 
startState =[6,4];
goalState =[1,9];

% obstacle 
global obstacle1 obstacle2
obstacle1={};
for ii=1:8
    obstacle1{end+1} = [4,ii];
end
obstacle2 ={};
for ii=2:9
    obstacle2{end+1}= [4,ii];
end

% for the environment dimension 
global height width
height = 6;
width =9;

global epsilon gamma alpha kapa 
epsilon = .1; % epsilon greedy 
gamma = .95; % discount 

% since this is not a static model. this parameter 
% will influence the performance significantly 
% tune this alpha \in [.1,.9]
alpha = .4; % model learning update step size 
kapa =1e-4; % for the exploring bonus parameter

global modelUpdateSteps
% this parameter will greatly affect the final cumulative 
% rewards. The larger the final cumulative reward will be. 
modelUpdateSteps = 50;

% the parameters \alpha and modelUpdateSteps are 
% manually selected, I found it best match the figure in the book. 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [reward, nextState] = env(state,action,maxSteps)
% this function simulates the environment
% current state and action as the input 
% reward and next state as the output 

% attention, this should be reset to zero after exceed maximal 
% time steps 
persistent visitimes  
% declear persistent variable to keep track the time elapse 
if isempty(visitimes)
    visitimes=0; 
end
visitimes = visitimes+1;
% reset 


global startState goalState 
global right left up down actions
global obstacle1 obstacle2 height width 

reward =0;
nextState=[0,0];
% check whether is the goalState 
if state == goalState
    reward = 0;
    next_state = goalState;
    return;
end

if visitimes <=1000
    obstacle = obstacle1;
else 
    obstacle = obstacle2;
end

% reset after choose obstacle 
if visitimes >= maxSteps
    visitimes = 0;
end

% check the action 
if ~ismember(action,actions)
    fprintf('No this action %d\n',action);
    error();
    return;
end

% check the validity of the state 
if state(1)<1 || state(1)>height || ...
        state(2)<0 || state(2)>width
    fprintf('No this stats (%d,%d)\n',state(1),state(2));
    error();
    return;
end

% if state on the obstacle 
for ii = 1:length(obstacle)
    if isequal(state,obstacle{ii})
        if isequal(state,[4,9]) 
           % if the state just on the obstacle, this happens when change
           % from obstacle1 to obstacle two, and state = [4,9];
           % since this rarely happen, it is fine to let state=[5,9];
            state = [5,9]; 
            
        else
            fprintf('This state does not exist\n');
            error();
            return;
        end
    end
end


% determin next state 
nextState =state;
if action ==up 
    nextState(1) = max(state(1)-1,1);
elseif action ==down 
    nextState(1) = min(state(1)+1,height);
elseif action == right
    nextState(2) = min(state(2)+1,width);
elseif action == left
    nextState(2) = max(state(2)-1,1);  
end

% determin the reward 
reward = 0;

if isequal(nextState, goalState) % if is goalState, then reward = 1;
    reward = 1;
end

for ii = 1:length(obstacle)
    if isequal(nextState,obstacle{ii})
        nextState = state;
        break;
    end
end

function [rewards,firstvisitime] = DynaQ(runTimes)
% implementes the DynaQ algorithm
global height width 
global actions up left right down 
global epsilon gamma alpha
global startState goalState; 
global modelUpdateSteps 
Qsa = zeros(height,width,length(actions));
currentState = startState;

firstvisitime=3000;

runs = 0;
rewards = zeros(runTimes,1);
stateSeen = zeros(height,width);
stateActionsSeen=zeros(height,width,length(actions));
modelQR = zeros(height,width,length(actions));
modelQS = cell(height,width,length(actions));
first =1;
while(runs<runTimes)
    runs = runs+1;
    
    % epsilong gready 
    if rand(1)<1-epsilon
        % instead of return the first max indice
        % randomly return the max index if has multiple 
        % max. Thus expedite the random search at the begining 
         ma= max(Qsa(currentState(1),...
            currentState(2),:));
         ind = find(Qsa(currentState(1),...
            currentState(2),:)==ma);
         currentAction = ind(randi(length(ind),1));
    else 
        currentAction = actions(randi(4));
    end
%    currentAction 
    % execute action 
    [reward,nextState] = env(currentState,currentAction,runTimes);
%      fprintf('current State is (%d,%d)nextState is (%d,%d)\n',...
%          currentState(1),currentState(2),nextState(1),nextState(2));
    rewards(runs) = reward;
    % update Q function 
    Qsa(currentState(1),currentState(2),currentAction)=...
        (1-alpha)*Qsa(currentState(1),currentState(2),currentAction)...
        +alpha*(reward...
        +gamma*max(Qsa(nextState(1),nextState(2),:)));
    
    % update the model learning part
    stateSeen(currentState(1),currentState(2))=1;
    stateActionsSeen(currentState(1),currentState(2),currentAction)=1;
%     modelStateActionSeen(currentState(1),currentState(2),currentAction) = 1;
    modelQR(currentState(1),currentState(2),currentAction)=reward;
    modelQS{currentState(1),currentState(2),currentAction} = nextState;
    
    for ii=1:modelUpdateSteps
        % randomly choose one previously seen state and action
        % find all have seen 
%         ts = find(modelStateActionSeen~=0);
%         % randomly choose one 
%         ts = ts(randi(length(ts),1));
%         [sampleState(1),sampleState(2),sampleAction]=ind2sub([height,width,length(actions)],ts);
        % select state 
        ts = find(stateSeen~=0);
        ts = ts(randi(length(ts),1));
        [sampleState(1),sampleState(2)] =ind2sub([height,width],ts);
      
        % select action 
        ts = find(stateActionsSeen(sampleState(1),sampleState(2),:)~=0);
        sampleAction = ts(randi(length(ts),1));
        
        sampleReward = modelQR(sampleState(1),sampleState(2),sampleAction);
        sampleNextState = modelQS{sampleState(1),sampleState(2),sampleAction};
        Qsa(sampleState(1),sampleState(2),sampleAction) = ...
            (1-alpha)*Qsa(sampleState(1),sampleState(2),sampleAction)...
            +alpha*(sampleReward...
            +gamma*max(Qsa(sampleNextState(1),sampleNextState(2),:)));
        
    end
    
    % if terminate, start again 
    if isequal(nextState,goalState)
       currentState = startState;
       if first 
       firstvisitime = runs;
%        fprintf('run %d times\n',runs);
       first =0;
       end
    else
       currentState = nextState; 
    end
    
end



function [rewards,firstvisitime] = DynaQplus(runTimes)
% for test the Q plut algorithm 
% implementes the DynaQ algorithm
global height width 
global actions up left right down 
global epsilon gamma alpha kapa
global startState goalState; 
global modelUpdateSteps 
Qsa = zeros(height,width,length(actions));
currentState = startState;

firstvisitime=3000;

runs = 0;
rewards = zeros(runTimes,1);
stateSeen = zeros(height,width);
stateActionsSeen=zeros(height,width,length(actions));
% keep track of time last experience that state action 
SAtime = zeros(height,width,length(actions)); 
modelQR = zeros(height,width,length(actions));
modelQS = cell(height,width,length(actions));
first =1;
while(runs<runTimes)
    runs = runs+1;
    
    % epsilong gready 
    if rand(1)<1-epsilon
        % instead of return the first max indice
        % randomly return the max index if has multiple 
        % max. Thus expedite the random search at the begining 
         ma= max(Qsa(currentState(1),...
            currentState(2),:));
         ind = find(Qsa(currentState(1),...
            currentState(2),:)==ma);
         currentAction = ind(randi(length(ind),1));
    else 
        currentAction = actions(randi(4));
    end
%    currentAction 
    % execute action 
    [reward,nextState] = env(currentState,currentAction,runTimes);
%      fprintf('current State is (%d,%d)nextState is (%d,%d)\n',...
%          currentState(1),currentState(2),nextState(1),nextState(2));
    rewards(runs) = reward;
    % update Q function 
    Qsa(currentState(1),currentState(2),currentAction)=...
        (1-alpha)*Qsa(currentState(1),currentState(2),currentAction)...
        +alpha*(reward...
        +gamma*max(Qsa(nextState(1),nextState(2),:)));
    
    % update the model learning part
    
    
%     modelStateActionSeen(currentState(1),currentState(2),currentAction) = 1;
    
    
    
    % if first visit allow reward = 0, time =1;
    
    % see references https://instructure-uploads.s3.amazonaws.com/account_290000000098865/attachments/69184293/Chapter8.pdf?response-content-disposition=attachment%3B%20filename%3D%22Chapter8.pdf%22%3B%20filename%2A%3DUTF-8%27%27Chapter8.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJFNFXH2V2O7RPCAA%2F20171110%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20171110T224828Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=01eb65819e06c5ef63e94d20ce5dd0a2e5b678b06b215205900bb6baeeeeb8be page 36 
    if ~stateSeen(currentState(1),currentState(2)) % if first seen set the time = 1 ,reward = 1;
        stateSeen(currentState(1),currentState(2))=1;
        for ii=actions
            if ii==currentAction
                modelQR(currentState(1),currentState(2),ii)=reward;
                modelQS{currentState(1),currentState(2),ii} = nextState;
               % keep the visit time in the real interaction 
                SAtime(currentState(1),currentState(2),ii) = runs;
            else
                modelQR(currentState(1),currentState(2),ii)=0;
                modelQS{currentState(1),currentState(2),ii} = currentState;
               % keep the visit time in the real interaction 
                SAtime(currentState(1),currentState(2),ii) = 1;
            end
            
        end
    else
        modelQR(currentState(1),currentState(2),currentAction)=reward;
        modelQS{currentState(1),currentState(2),currentAction} = nextState;
        % keep the visit time in the real interaction 
        SAtime(currentState(1),currentState(2),currentAction) = runs;
    end
        

    
    for ii=1:modelUpdateSteps
        % randomly choose one previously seen state and action
        % find all have seen 
%         ts = find(modelStateActionSeen~=0);
%         % randomly choose one 
%         ts = ts(randi(length(ts),1));
%         [sampleState(1),sampleState(2),sampleAction]=ind2sub([height,width,length(actions)],ts);
        % select state 
        ts = find(stateSeen~=0);
        ts = ts(randi(length(ts),1));
        [sampleState(1),sampleState(2)] =ind2sub([height,width],ts);
      % this part is different from DynaQ algorithm 
        % select action 
%         ts = find(stateActionsSeen(sampleState(1),sampleState(2),:)~=0);
%         sampleAction = ts(randi(length(ts),1));
         sampleAction = randi(length(actions),1);
        
        % keep the visit time 
         tao = runs - SAtime(sampleState(1),sampleState(2),sampleAction);
%         SAtime(sampleState(1),sampleState(2),sampleAction) = runs; move
%         to the real interaction part 
%         tao 
        sampleReward = modelQR(sampleState(1),sampleState(2),sampleAction);
        sampleNextState = modelQS{sampleState(1),sampleState(2),sampleAction};
        Qsa(sampleState(1),sampleState(2),sampleAction) = ...
            (1-alpha)*Qsa(sampleState(1),sampleState(2),sampleAction)...
            +alpha*(sampleReward...
            +kapa*sqrt(tao)... % bonus for exploration 
            +gamma*max(Qsa(sampleNextState(1),sampleNextState(2),:)));
        
    end
    
    % if terminate, start again 
    if isequal(nextState,goalState)
       currentState = startState;
       if first 
       firstvisitime = runs;
%        fprintf('run %d times\n',runs);
       first =0;
       end
    else
       currentState = nextState; 
    end
    
end

function plot8_5()
runs = 10;
steps = 3000;
rewardsQplus = zeros(runs,steps);
rewardsQ = zeros(runs,steps);
firstvisitime = zeros(runs,1);
for ii=1:runs
    ii
    [rewardsQplus(ii,:),firstvisitime(ii)] = DynaQplus(steps);
    [rewardsQ(ii,:),~] = DynaQ(steps);
end
rewardsQplus = rewardsQplus*triu(ones(steps,steps),0);
rewardsQ = rewardsQ*triu(ones(steps,steps),0);
figure();
plot(mean(rewardsQ),'r','linewidth',2);
hold on;
plot(mean(rewardsQplus),'linewidth',2);

legend('DynaQ','DynaQ+');
xlabel('Time steps');
ylabel('Cumulative reward');
