function main(problem)
global Hit
Hit = 1;
global Strike
Strike = 0;
% define action 
timesRun=100;
nEps=10000;
if strcmp(problem,'prob2')
    plotMSE(timesRun, nEps);
elseif strcmp(problem,'prob1')
    plot_StateValue(5000000);
else fprintf('error\n');
end

function plot_StateValue(nEps)
% For Fig(5.1). 
% MC method to run 
% first entry for the none-usable ace
% second entry for the usable ace 
StateValue ={zeros(10,10),zeros(10,10)}; 
StateValue1 ={zeros(10,10),zeros(10,10)};
counts = {zeros(10,10),zeros(10,10)};
counts1 = {zeros(10,10),zeros(10,10)};

for ii=1:nEps % episodes
    [reward, state,~] = playThegame('onPolicy');

    if state(1)==0
        StateValue{1}(state(2)-11,state(3))=...
            StateValue{1}(state(2)-11,state(3))+reward;
        counts{1}(state(2)-11,state(3)) =...
            counts{1}(state(2)-11,state(3))+1;
    else
        StateValue{2}(state(2)-11,state(3))=...
            StateValue{2}(state(2)-11,state(3))+reward;
        counts{2}(state(2)-11,state(3)) =...
            counts{2}(state(2)-11,state(3))+1;
    end
    if ii==10000 % record at the episodes 10000
        StateValue1{1} = StateValue{1}./counts{1};
        StateValue1{2} = StateValue{2}./counts{2};
    end       
end
StateValue{1} = StateValue{1}./counts{1};
StateValue{2} = StateValue{2}./counts{2};
plotStateValue(StateValue1);
plotStateValue(StateValue);

% plot the figure;
function plotStateValue(StateValue)
figure();
[X,Y]=meshgrid(12:21,1:10);
Z = StateValue{1};
surf(X,Y,Z');
title('Non Usable Ace');
xlabel('Sum of Player Cards');
ylabel('Dealers Cards');


figure();
Z = StateValue{2};
surf(X,Y,Z');
xlabel('Sum of Player Cards');
ylabel('Dealers Cards');
title('Usable Ace');



function [rewards,ratio]=importanceSampling(nEps)
  rewards = zeros(nEps,1);
  ratio = zeros(nEps,1);
  % init state, usable ace and playersum=13 and dealer first card = 2
  StateInit = [1,13,2];
  for ii=1:nEps     
  [reward, ~,stateTrajectory] = playThegame('offPolicy',StateInit);
  rewards(ii) = reward;
  ratio_on = 1;
  ratio_off=1;
    for jj=1:length(stateTrajectory)
        playersum = stateTrajectory{jj}(2);
        action   = stateTrajectory{jj}(4);
        % if does on plicy probability is zero, then break
        if PlayerOnPolicy(playersum)~=action 
            ratio_on = 0; 
            break;
        else
            ratio_off = ratio_off*.5;
        end
     
     end
   ratio(ii) = ratio_on/ratio_off;  
   
  end
% after get all the   
function [reward, state,stateTrajectory] = playThegame(actionName,StateInit)
% State=[PlayerAce,playerSum,DealeFirstCard]
global Hit
global Strike

if strcmp(actionName,'onPolicy') % for the player, use which kind of policy 
    PlayerPolicy = @PlayerOnPolicy;
else PlayerPolicy = @PlayerOffPolicy;
end
% whether given the inite state of player or Dealer 
if nargin ==1 % if only input the policy then init the state 
              % of the player and dealer 
   PlayerSum = 0;
   PlayerAce = 0; % number of Aces 
   while(PlayerSum<12) % while exceed 11, else hit 
       card = getCard();
       if card ==1 % A  
          PlayerSum = PlayerSum + 11; % in the init state, always ace always
                                      % usable
          PlayerAce =PlayerAce + 1;
       else
          PlayerSum = PlayerSum+card;
       end
   end
   if PlayerSum >21
       PlayerSum = PlayerSum-10; % use one ace as 1 
       PlayerAce = PlayerAce -1; 
   end
   
   DealerHideCard = getCard();
   DealerFirstCard = getCard();
   
   
else
   PlayerAce = StateInit(1);
   PlayerSum = StateInit(2);
   DealerFirstCard = StateInit(3);
   DealerHideCard =getCard();   
end

% the initialize state is done here
% playing;
% first player play
stateTrajectory =[];
state = [PlayerAce,PlayerSum,DealerFirstCard];
while(1)
   action = PlayerPolicy(PlayerSum);
   stateTrajectory{end+1}= [PlayerAce,PlayerSum,DealerFirstCard,action]; 
   if action== Strike
       break;
   else
       card = getCard();
%        if card == 1  should never use 11, because state>12 
%            PlayerAce = PlayerAce + 1;
%            PlayerSum = PlayerSum + 11;
%        else 
           PlayerSum = PlayerSum+card;
%        end
       
   end
   if (PlayerSum >21) % if bust 
       if (PlayerAce >0) % if use ace, then not use it 
           PlayerSum = PlayerSum - 10;
           PlayerAce = PlayerAce - 1;
       else % blust           
           reward = -1;
           return;
       end   
   end   
    
end

% dealer's turn 
if (DealerFirstCard ==1) && (DealerHideCard ==1)
    DealerSum = 11+1;
    DealerAce = 1;
elseif (DealerFirstCard ==1) && (DealerHideCard~=1)
    DealerSum = 11+DealerHideCard;
    DealerAce = 1;
elseif DealerFirstCard~=1 && DealerHideCard ==1
    DealerSum = DealerFirstCard+11;
    DealerAce = 1;
else
    DealerSum = DealerFirstCard+DealerHideCard;
    DealerAce = 0;
end

% dealer hit or strike 
while(1)
  action = DealerPolicy(DealerSum);
  if action == Strike;
      break;
  else 
     card = getCard(); 
     % there is some ambiguity for the strategy of dealer
%       if card == 1 % used as 1 is the same as use as 11 
%           DealerSum = DealerSum + 11;
%           DealerAce = DealerAce + 1;
%       else
         DealerSum = DealerSum+card;
%       end
  end
  if DealerSum > 21 
      if DealerAce > 0 
          DealerSum = DealerSum - 10;
          DealerAce = DealerAce - 1;
      else
          reward = 1; % dealer bust, you win 
          return;
      end
  end
end

% if both strike, then compare 
if DealerSum > PlayerSum
    reward = -1;
elseif DealerSum < PlayerSum
    reward = 1;
else
    reward = 0;
end
return; 

function action = PlayerOnPolicy(sumOfnum)
% player action only depends on the sum of the numbers 
global Hit
global Strike
if sumOfnum<20 % 
    action = Hit;
else
    action = Strike;
end

function action = DealerPolicy(sumOfnum)
% also only depends on sum of the number 
global Hit
global Strike
if sumOfnum>=17 
    action = Strike;
else action = Hit;
end

function action = PlayerOffPolicy(sumOfnum)
global Hit
global Strike
temp = randn(1);
if temp > 0 
    action = Hit; 
else action = Strike;
end

function cardNum= getCard()
% from A~K which is 1~13
temp = randi(13,1); 
cardNum = min(temp,10);

function plotMSE(timesRun, nEps)
% timesRun
true_value = -.27726;
Ord_Est = zeros(timesRun,nEps);
Wei_Est = zeros(timesRun,nEps);
for ii=1:timesRun
    [rewards,ratio] = importanceSampling(nEps);
        
    % weighted importance sampling 
    Cn = 0;
    y_ = 0;
    y_o = 0; 
    for jj=1:nEps % incremental implement of weighted importance sampling
        Cn = Cn+ ratio(jj);
        if Cn>0 % to avoid divided by zero 
           y = y_ + ratio(jj)/Cn*(rewards(jj)-y_);
           y_ = y;
           Wei_Est(ii,jj) = y_;
        end
        Ord_Est(ii,jj) = y_o+(rewards(jj)*ratio(jj)-y_o)/jj;
        y_o = Ord_Est(ii,jj);
    end
        
end
% means square error 
mse_ord = mean((Ord_Est - true_value).^2);
mse_wei = mean((Wei_Est - true_value).^2);
figure();
semilogx(1:nEps,mse_ord,'g','linewidth',1.5);
hold on;
semilogx(1:nEps,mse_wei,'r','linewidth',1.5);
xlabel('Episodes (log scale)');
ylabel('Mean Sequare Error');
legend('Ordinary Importance Sampling','Weighted Importance Sampling');
ylim([0,4]);
