% number of bandits
N =10;
% the reward expectation 

Reward_greedy = zeros(2000,1000);
for ii=1:2000 % for 2000 runs 
    q = randn(10,1);
    ii; 
    % epsilon greedy one run 
    % initialize
    % action value 
    Qa = zeros(N,1);
    % number of action 
    Na = zeros(N,1);
    
    % epsilon greedy
    eps = .1;
    iter = 0;
    max_itr = 1000;
    while(iter < max_itr )
        iter = iter +1;
        % espilon greedy strategy 
        temp = rand(1);
        if temp <1-eps
            [val,A] = max(Qa);
            % randomly choose the optimal index
            ind = find(Qa==val);
            A = ind(randi(length(ind)),1);
        else A = randi(N,1);
        end
        % reward 
        R = random('norm',q(A),1);
        Na(A) = Na(A)+1;
        Qa(A) = Qa(A)+1/Na(A)*(R-Qa(A));
        Reward_greedy(ii,iter) = R;
    end
end


%% UCB methods 
c = 2;
Reward_UCB=zeros(2000,1000);

for ii=1:2000
    ii;
    q = randn(10,1);
    Qa = zeros(N,1);
    Na = zeros(N,1);
    iter = 0;
    
    while(iter < 1000)
        iter = iter +1;
        ActQ = Qa+c*sqrt(log(iter+10^(-15))./Na); % to avoid 0/0
        [val,A]=max(ActQ);
        ind = find(ActQ==val);
        % randomly choose the maximal if
        % there are more than one optimal action 
        A = ind(randi(length(ind)),1);
        % reward 
        R = random('norm',q(A),1);
        Na(A) = Na(A)+1;
        Qa(A) = Qa(A)+1/Na(A)*(R-Qa(A));
        Reward_UCB(ii,iter) = R;
    end
end

plot(mean(Reward_greedy));
hold on;
plot(mean(Reward_UCB),'r-');
xlabel('steps');
ylabel('Average Reward');
legend('Greedy \epsilon=0.1','UCB c=2');
