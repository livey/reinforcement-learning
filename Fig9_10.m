function main()
close all;
dbstop iff error 
global trueStateValues
% test block 
init();
plot9_10();

function init()
global Left Right actions maxSteps
Left = -1;
Right = 1;
actions =[Left, Right];
maxSteps = 100; % reduce the computation 


global numOfStates trueStateValues
numOfStates = 1000; % except the terminate states 

% the new stateValue evaluation 
A = zeros(numOfStates);
b = zeros(numOfStates,1);
for ii=1:numOfStates
    state = ii;
    rewards = 0;
    for jj=1:maxSteps
        newState = min(state+jj,1001);
        if newState == 1001;
            rewards=rewards+1;
        else
            A(ii,newState) = .5/maxSteps;
        end
        
    end
    for jj=1:maxSteps
        newState = max(state-jj,0);
        if newState == 0;
            rewards=rewards-1;
        else
            A(ii,newState) = .5/maxSteps; 
        end
        
    end
   b(state) = rewards/2/maxSteps;     
end
trueStateValues = inv(eye(numOfStates)-A)*b;

global startState termiStates
startState = 500; % it is equivalent to 501; 
termiStates = [0,numOfStates+1];
global gamma alphaSingle alphaMulti
gamma = 1; % discount 
alphaSingle = .00034; % update step size 
alphaMulti  = .0005/50;

function indx = state2sub(state,type)
% input all the states 
% to speed up call 
% convert state to index 
% for multiple 50 tilings, 
% index1 in [1,50];  
% index2 in [1,6]
% so the params has dimention 50 \times 6 
tileSize = 200;
groups = 50;
offSet = 4;
numOftiles = 6; % large 5+1;
if strcmp(type,'single')
    indx = fix((state-1)/tileSize)+1;
elseif strcmp(type,'multi')
    indx = zeros(groups,length(state));
    for ii=1:groups
       indx(ii,:) = fix((state+200-(ii)*offSet)./tileSize)+1; 
    end
else
    error('no such coding type');
end

function [trajectory,reward] = play()
global Left Right actions 
global startState  termiStates
global maxSteps 
trajectory = [];
currentState = startState;
trajectory(end+1) = currentState;
while( (currentState ~=termiStates(1))&& (currentState~=termiStates(2)))
    action = actions(randi(2)); % randomly select an action
    steps = randi(maxSteps,1);
    if action == Left
        nextState = max(currentState-steps,0);
    else
        nextState = min(currentState +steps,termiStates(2));
    end
    currentState = nextState;
    trajectory(end+1) = currentState;
end

if currentState == 0
    reward = -1;
else 
    reward = 1;
end

function rmse = singleTile(episodes)
global numOfStates alphaSingle trueStateValues
stateValues = zeros(numOfStates,1);
w = zeros(5,1);
rmse = zeros(episodes,1);
for kk = 1:episodes
%     fprintf('runing single tile episodes %d\n',kk);
    [tra,r]=play();
    subs = state2sub(tra(1:end-1),'single'); % batch convert state to subscripts
    for ii=1:length(tra)-1
        sub = subs(ii);
        wt = w(sub);
        wt = wt + alphaSingle*(r - wt)*1;
        w(sub) = wt;
    end
 
    stateValues = w(state2sub(1:1000,'single'));
    rmse(kk) = sqrt(mean((stateValues-trueStateValues).^2));
end

function rmse = multiTile(episodes)
global numOfStates alphaSingle trueStateValues
global alphaMulti 
stateValues = zeros(numOfStates,1);
w = zeros(50,6);
rmse = zeros(episodes,1);
for kk = 1:episodes
%     fprintf('runing multi tile episodes %d\n',kk);
    [tra,r]=play();
    subs = state2sub(tra(1:end-1),'multi');
    for ii=1:length(tra)-1
        sub = subs(:,ii);
        indx = (sub-1)*50+[1:50]';
        wt = w(indx);
        wt = wt + alphaMulti*(r - sum(wt));
        w(indx) = wt;
    end
    
    subs = state2sub(1:1000,'multi');
    for ii = 1:numOfStates
        sub = subs(:,ii);
        indx= (sub-1)*50+[1:50]';
        stateValues(ii) = sum(w(indx));
    end
    rmse(kk) = sqrt(mean((stateValues-trueStateValues).^2));
end


function plot9_10()
runs = 100;
episodes = 5000;
RMSVEsingle =zeros(runs,episodes); 
RMSVEmulti  =zeros(runs,episodes);
for ii=1:runs 
    fprintf('run %d times\n',ii);
    RMSVEsingle(ii,:) = singleTile(episodes)';
    RMSVEmulti(ii,:) = multiTile(episodes)';
end

plot(mean(RMSVEsingle),'r','linewidth',1.5);
hold on;
plot(mean(RMSVEmulti),'linewidth',1.5);
xlabel('Episodes');
ylabel('RMSVE');
ylim([0,0.4]);
legend('State aggregation (one tiling)',...
    'Tile coding (50 tilings)');
