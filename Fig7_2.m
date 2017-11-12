function main()
clear all;
dbstop iff error % for debug 
%global SAS SAR % for state action state, state action reward
global numOfstates numOfactions Right Left
global startState termiState realStateValue GAMMA
GAMMA =1;
numOfstates = 21;
startState = 11;
termiState = [1,numOfstates];
numOfactions = 2;
Right =1;
Left =2;

% compute the real state value 
% temporarily compute state 2~18 value 
% by solving bellman equation v=Av+b;
A= zeros(numOfstates-2,numOfstates-2);
b = zeros(numOfstates-2,1);
realStateValue = zeros(numOfstates,1);
for ii=2:numOfstates-3
    A(ii,ii-1) = .5*GAMMA;
    A(ii,ii+1) = .5*GAMMA;
end
% for state 2
A(1,2) = .5*GAMMA;
b(1) =-.5;
A(numOfstates-2,numOfstates-3) = .5;
b(numOfstates-2) = .5;
realStateValue(2:numOfstates-1) = inv(eye(numOfstates-2)-A)*b;
realStateValue(1)=0;
realStateValue(numOfstates)=0;

% for test 
%    sv =zeros(numOfstates,1);
%   for ii=1:100
%    sv=TDn(sv,.2,1)
%   end
%    sv

% plot fig_7.2
   fig_7_2();

function stateValue = TDn(stateValue,alpha,nStep)
global startState termiState GAMMA 
currentState = startState;
states = currentState;
time = 0;
T = inf;
rewards = [];
while(1)
     time = time+1;
     if time<T
         if randn() <0
             nextState = currentState -1;
         else
             nextState = currentState+1;
         end
         if nextState == termiState(1)
             reward = -1;
             T = time;
         elseif nextState == termiState(2)
             reward = 1;
             T = time;
         else reward =0;
         end
         
         states(end+1) = nextState;
         rewards(end+1) = reward;
     end
         updateTime = time-nStep+1;
         % for index convenience 
         % start from index 1 
     if updateTime >=1 
         G =0 ;
        for ii=updateTime:min(T,updateTime+nStep-1)
            G =GAMMA^(ii-updateTime)*rewards(ii)+G;
        end
        if updateTime+nStep <=T 
            G = G+GAMMA^nStep*stateValue(states(updateTime+nStep));
        end

        stateValue(states(updateTime)) = stateValue(states(updateTime))...
            +alpha*(G-stateValue(states(updateTime)));
     end
     if updateTime == T
         break;
     end
     currentState = nextState;
end

function fig_7_2()
global numOfstates
global realStateValue
episodes = 10;
repeats = 100;
nSteps = 2.^(0:9);
alphas =0.01:0.01:1;
errors = zeros(length(alphas),length(nSteps));
for alphai =1:length(alphas)
    alpha = alphas(alphai);
    for ni =1:length(nSteps)
        n = nSteps(ni);
        fprintf('alpha = %4f, n=%d\n',alpha,n);
        ter = zeros(length(repeats),1);
        for ii = 1:repeats
            % init state value
            stateValue =zeros(numOfstates,1);
            for jj = 1:episodes
                stateValue = TDn(stateValue,alpha,n);
            end
            ter(ii) = sqrt(mean((stateValue-realStateValue).^2));
        end
        errors(alphai,ni) = mean(ter);
    end
end
plot(alphas,errors,'linewidth',1);
ylim([min(errors(:)-.1),.55]);
legend('n=1','n=2','n=4','n=8','n=16','n=32','n=64','n=128',...
    'n=256','n=512');
xlabel('\alpha');
ylabel('RMS');
