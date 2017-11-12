function main()
%% Macro 

params.MAX_CARS = 20;
params.MAX_MOVE_CARS = 5;

% the lambda value is re-mapped to the index 
params.FIRST_REQUEST_LAMBDA =2;
params.SECOND_REQUEST_LAMBDA = 3;

params.FIRST_RETURN_LAMBDA =2;
params.SECOND_RETURN_LAMBDA = 1;

% positive from first to second 
params.FREE_NUMBER_CARS=1; %%
params.COST_PER_CAR = 2;
params.GAMMA = 0.9; 
params.PARK_FREE_NUM=10;  %%
params.PARK_FEE = 4;      %%
params.RENT_CREDIT = 10;

% construct a poisson distrubtion matrix to boost the computation 
poisson = zeros(params.MAX_CARS+1,3); % column represents the lambda params 2,3,4
poisson(:,1) =poisspdf(0:params.MAX_CARS,2);
poisson(:,2) =poisspdf(0:params.MAX_CARS,3);
poisson(:,3) =poisspdf(0:params.MAX_CARS,4);
params.POISSON = poisson;
% test performace 


stateValue=zeros(params.MAX_CARS+1,params.MAX_CARS+1);
expect_return([0,0],0,stateValue,params)


policy = zeros(params.MAX_CARS+1,params.MAX_CARS+1);

actions = [-params.MAX_MOVE_CARS:params.MAX_MOVE_CARS];
improvePolicy =0;
policyHist={};
stateValueHist={};
policyHist{end+1} = policy;

while(1)
        fprintf('State Value Update\n');
%         newStateValue= stateValue;
        
        % for state value evaluation 
        Delt = 0;
        for ii=0:params.MAX_CARS
            for jj=0:params.MAX_CARS
%                 fprintf('Update State Value(%d,%d)\n',ii,jj);
%                 newStateValue(ii+1,jj+1) = expect_return([ii,jj],policy(ii+1,jj+1),stateValue,params);       
                temp = expect_return([ii,jj],policy(ii+1,jj+1),stateValue,params);
                Delt = max(Delt,abs(temp-stateValue(ii+1,jj+1)));
                % update the stateValue immediately 
                stateValue(ii+1,jj+1) = temp; 
            end
        end
        Delt 
        %if norm(newStateValue-stateValue,'fro')<10^-3
        if Delt < 1e-3
           fprintf('State Value is Stable\n');
           improvePolicy = 1;
           stateValueHist{end+1} = stateValue;
        end
%         stateValue = newStateValue;
        
       % policy improvement  
        if improvePolicy 
           fprintf('Improve policy');
           newPolicy = zeros(params.MAX_CARS+1,params.MAX_CARS+1);
           for ii=0:params.MAX_CARS
              for jj=0:params.MAX_CARS
                  fprintf('Update policy at state (%d,%d)\n',ii,jj);
                  actionReturns =[];
                  for action = actions
                      if (action>=0 && ii>=action) || (action<0 && jj>=abs(action))
                           actionReturns = [actionReturns,expect_return([ii,jj], action, stateValue,params)];
                      else actionReturns = [actionReturns,-inf];      
                      end                    
                  end
                  
                  [~,bestAction] = max(actionReturns); 
                  newPolicy(ii+1,jj+1) = actions(bestAction);
              end
           end
           policyChangd = sum(newPolicy~=policy);
           
           if policyChangd ==0
               fprintf('Policy is stable, exit\n');
               policy = newPolicy;
 %              policyHist{end+1} = policy;
               break;
           end
           policy = newPolicy;
           policyHist{end+1} = policy;
           improvePolicy =0;
        end
        
end

%% plot 
figure();
[X,Y]=meshgrid(0:params.MAX_CARS);
surf(X,Y,stateValue);
xlabel('num of cars at second location');
ylabel('num of cars at first location');
zlabel('state value');

for ii=1:length(stateValueHist)-1
figure();
imagesc(policyHist{ii});
colorbar;
xlabel('num of cars at second location');
ylabel('num of cars at first location');
title(['\pi', num2str(ii)]);
end
figure();
imagesc(policyHist{ii+1});
colorbar;
xlabel('num of cars at second location');
ylabel('num of cars at first location');
title(['\pi_*']);
save allData

function expt = expect_return(state,action,stateValue,params)
%% action \in (-Max_car,max_car),
  % check whether action is valid 
  if action > params.MAX_MOVE_CARS || action<-params.MAX_MOVE_CARS
      expt = -inf;
      return;
  end
  
  poisson = params.POISSON;
  expt = 0;
  % the cost for moving 
  if params.FREE_NUMBER_CARS>=0
      if action>=params.FREE_NUMBER_CARS
          expt = expt -params.COST_PER_CAR*(action-params.FREE_NUMBER_CARS);
      elseif action <0
          expt =expt +action*params.COST_PER_CAR;

      else
          expt = 0;
      end
  else
      if action<=params.FREE_NUMBER_CARS
          expt = expt + params.COST_PER_CAR*(action+params.FREE_NUMBER_CARS);
      elseif action >=0
          expt = expt - params.COST_PER_CAR*(action);
      else expt = 0;
      end
  end
    
  % the cost for parking 
   numOfCarsFirstFix  = min(state(1)-action,params.MAX_CARS);
   numOfCarsSecondFix = min(state(2)+action,params.MAX_CARS);
  
  if numOfCarsFirstFix > params.PARK_FREE_NUM
      expt = expt - params.PARK_FEE;
  end
  if numOfCarsSecondFix > params.PARK_FREE_NUM
      expt = expt -params.PARK_FEE;
  end

  for first_request_num=0:params.MAX_CARS
      pb1=poisson(first_request_num+1,params.FIRST_REQUEST_LAMBDA);
      for second_request_num=0:params.MAX_CARS
%           numOfCarsFirst  = min(state(1)-action,params.MAX_CARS);
%           numOfCarsSecond = min(state(2)+action,params.MAX_CARS);
          pb2=poisson(second_request_num+1,params.SECOND_REQUEST_LAMBDA);
          reward = 0;
          % valid request 
          real_rent_first = min(first_request_num,numOfCarsFirstFix);
          real_rent_second = min(second_request_num,numOfCarsSecondFix);
          % reward for rentaling cars 
          reward =  reward + (real_rent_first+real_rent_second)*params.RENT_CREDIT;
          
          numOfCarsFirst = numOfCarsFirstFix - real_rent_first;
          numOfCarsSecond = numOfCarsSecondFix - real_rent_second;
          
          for returnFirst = 0:params.MAX_CARS
              pb3=poisson(returnFirst+1,params.FIRST_RETURN_LAMBDA);
              for returnSecond = 0:params.MAX_CARS
                  pb4=poisson(returnSecond+1,params.SECOND_RETURN_LAMBDA);
                  numOfCarsFirst = min(numOfCarsFirst+returnFirst,params.MAX_CARS);
                  numOfCarsSecond = min(numOfCarsSecond+returnSecond,params.MAX_CARS);
                  prob = pb1*pb2*pb3*pb4;
                   % '+1' means index state value 
                   expt = expt+ prob*(reward+...
                       params.GAMMA*stateValue(numOfCarsFirst+1,numOfCarsSecond+1)); 
              end
          end
      end
  end
          
                   
end

