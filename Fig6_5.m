function q1_main()
% define globle action, state-action->reward, state-action->state
global HEIGHT  WIDTH  UPP DOWN LEFT RIGHT SAS Reward_sa 
global EPSILON GAMMA ALPHA NumOfActions StartState TermiState actions
EPSILON = .1; % for epsilon greedy 
GAMMA = 1; % for discount 
ALPHA = .5; % for updating step size 
HEIGHT = 4; % worl height  
WIDTH = 12; % world width 
UPP = 1; % actions 
DOWN = 2;
LEFT = 3;
RIGHT =4;
actions =[UPP,DOWN,LEFT,RIGHT]; % actions 
NumOfActions = 4;
StartState = [HEIGHT,1]; 
TermiState = [HEIGHT,WIDTH];
% reward for the state action pair 
Reward_sa = ones(HEIGHT,WIDTH,NumOfActions)*-1;
Reward_sa(3,2:WIDTH-1,DOWN)=-100;
Reward_sa(HEIGHT,1,RIGHT)=-100;
% state - action -> state table 
SAS = {HEIGHT,WIDTH,NumOfActions};

for ii=1:HEIGHT
    for jj=1:WIDTH
        SAS{ii,jj,UPP}  = [max(ii-1,1),jj];
        SAS{ii,jj,DOWN} = [min(ii+1,HEIGHT),jj];
        SAS{ii,jj,LEFT} = [ii,max(jj-1,1)];
        SAS{ii,jj,RIGHT}= [ii,min(jj+1,WIDTH)];
    end
end
% consider the cliff 
for ii=2:11
  SAS{3,ii,DOWN}= StartState;
end
SAS{HEIGHT,1,RIGHT} =StartState;

%% Start the homework program 
fig6_5();

function action = greedy_policy(QsaValue,state)
global EPSILON  
global actions 
tmp_val = zeros(4,1);
for ii = actions
    tmp_val(ii) = QsaValue(state(1),state(2),ii);
end
if rand(1)<=1-EPSILON  % greedy 
    [~,ind]=max(tmp_val);
    action = actions(ind);
    return;
else % epsilon random 
    action = actions(randi(length(actions))); 
end


function fig6_5()
global HEIGHT WIDTH NumOfActions
episodes = 500;
rewards_sarsa = zeros(episodes,1);
rewards_qlearning = zeros(episodes,1);

% do 20 independent runs, it is hard to get 
% the smoothed plot like in the book; 
runs = 40;
for jj = 1:runs
    jj 
Qsa_sarsa = zeros(HEIGHT,WIDTH,NumOfActions);
Qsa_qlearning = zeros(HEIGHT,WIDTH,NumOfActions);
for ii=1:episodes
    [Qsa_sarsa,reward_sarsa] = sarsa(Qsa_sarsa);
    rewards_sarsa(ii) = rewards_sarsa(ii)+reward_sarsa;
    [Qsa_qlearning,reward_qlearning] = q_learning(Qsa_qlearning);
    rewards_qlearning(ii) = rewards_qlearning(ii)+reward_qlearning;
end

end
rewards_sarsa = rewards_sarsa/runs;
rewards_qlearning = rewards_qlearning/runs;

% do averaging 10 successive averaging 
ave_steps = 10;
for ii=1:episodes
    y_sarsa(ii)     = mean(rewards_sarsa(ii:min(ii+ave_steps,episodes)));
    y_qlearning(ii) = mean(rewards_qlearning(ii:min(ii+ave_steps,episodes)));
end
plot(y_sarsa,'linewidth',2);
hold on;
plot(y_qlearning,'r','linewidth',2);
legend('Sarsa','Q-learning');
ylim([-100,0]);
ylabel('Sum of rewards during episode');
xlabel('Episodes')

function [Qsa,reward] = sarsa(Qsa)
global StartState TermiState ALPHA GAMMA SAS Reward_sa
reward=0;
CurrentState = StartState;
CurrentAction = greedy_policy(Qsa,CurrentState);
while(~isequal(CurrentState,TermiState))
    % find the next state 
    NextState = SAS{CurrentState(1),CurrentState(2),CurrentAction};
    % next action 
    NextAction = greedy_policy(Qsa,NextState);
    % reward
    Rt = Reward_sa(CurrentState(1),CurrentState(2),CurrentAction);
     
    Qsa(CurrentState(1),CurrentState(2),CurrentAction)...
        =(1-ALPHA)*Qsa(CurrentState(1),CurrentState(2),CurrentAction)...
        +ALPHA*(Rt+GAMMA*Qsa(NextState(1),NextState(2),NextAction));
    reward = reward + Rt;
    CurrentState = NextState;
    CurrentAction = NextAction;
end


function [Qsa,reward] = q_learning(Qsa)
global StartState TermiState ALPHA GAMMA SAS Reward_sa
reward=0;
CurrentState = StartState;

while(~isequal(CurrentState,TermiState))
    % find the next state 
    CurrentAction = greedy_policy(Qsa,CurrentState);
    NextState = SAS{CurrentState(1),CurrentState(2),CurrentAction};
    % next action 
    
    % reward
    Rt = Reward_sa(CurrentState(1),CurrentState(2),CurrentAction);
     
    Qsa(CurrentState(1),CurrentState(2),CurrentAction)...
        =(1-ALPHA)*Qsa(CurrentState(1),CurrentState(2),CurrentAction)...
        +ALPHA*(Rt+GAMMA*max(Qsa(NextState(1),NextState(2),:)));
    reward = reward + Rt;
    CurrentState = NextState;
end
