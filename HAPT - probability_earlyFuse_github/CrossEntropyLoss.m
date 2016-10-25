% Copyright (c) Jun Ye. 2016. This program is free software: you can 
% redistribute it and/or modify it under the terms of the 
% GNU General Public License as published by the Free Software Foundation, 
% either version 3 of the License, or (at your option) any later version. 
% This program is distributed in the hope that it will be useful, 
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% 
% See the GNU General Public License for more details. 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see http://www.gnu.org/licenses/.

% training the FTA projections
function LearnedProjection = CrossEntropyLoss(NormProjection,...
                                 train, subsetIndex,...
                                 dim, K, learningRate,...
                                 threshold, alpha, ...
                                 lambda1, lambda2, maxEpoch,...
                                 positivePairSampleRate, ...
                                 negativePairSampleRate, ...
                                 timeDuration)
                             
LearnedProjection = NormProjection;
batchSize = size(subsetIndex, 1);
lastTrainError=1e10;
trainingError=1e9;
epoch=1;
 error=[];
% normm = [];
% LRate=[];

% subsampling
[trainCell, labelset, ~] = ...
    select_train_cell(train, batchSize, subsetIndex, timeDuration, dim);

lastGradW_delta=zeros(K,dim);
% terminal critera
%tstart = tic;
previousError = inf;

timee = tic;
while (epoch<=maxEpoch && ...
        abs((lastTrainError-trainingError)/lastTrainError)>1e-10)
    %fprintf('Epoch = %d \n', epoch);
    cnt=0;
%     cntp=0;
%     cntn=0;
    lastTrainError = trainingError;
    trainingError=0;    
    proj = mtimesx(LearnedProjection, trainCell);

    sum_gradW = zeros(K, dim);
    
    %sigmoid
    P = 1./(1+exp(-alpha*proj));  
%     P(P<1e-4)=0;
%     P(P>1-1e-4)=1;
    
    P = mtimesx(1,P,'t'); 
    h = ComputeLeadingProbMatrix(P, timeDuration, 1, K);  
    h = reshape(h, K+1, 1, []);
    
    % compute gradP(dim, T, batchSize, K)
    gradP = zeros(dim, timeDuration, batchSize, K);
    
    tmpP = mtimesx(1,alpha*P.*(1-P),'t'); 
    for k = 1:K
        gradP(:,:,:,k) = bsxfun(@times, trainCell, tmpP(k,:,:)); 
    end
    %gradM (dim, K+1, batchSize, K);  [h0 h1 h2 h3]  
    gradM = zeros(dim, K+1, batchSize, K);
    
    %% compute grad_h0
    prod_3K = prod(1-P);
    for k =1:K
        tail = ones(1,1,batchSize);
        for k1 = 1:K
            if k1~=k
                tail = tail.*prod_3K(:,k1,:);
            end
        end  
        tmp2=zeros(timeDuration,1,batchSize);
        for t = 1:timeDuration
            P_inv = 1-P(:,k,:);
            P_inv(t,:,:) = 1;
            tmp2(t,:,:) = prod(P_inv);
        end
        tmp3 = mtimesx(-gradP(:,:,:,k), tmp2);
        gradM(:,1,:,k)= mtimesx(tmp3, tail);
    end    
    
    %% compute grad_h1  ~ grad_h3 
    cprod3 = cumprod(1-P);
    cprodBefore = [ones(1, K, batchSize); cprod3(1:end-1,:, :)];
    
    for k1 = 1:K
        for k = 1:K
            if k==k1
                localcopy = cprodBefore;
                localcopy(:,k,:)=1;
                tail = prod(localcopy,2);
                P_inv = 1-P;
                P_inv_k_backup = P_inv(:,k,:);
                P_inv(:,k:end,:)=1;
                tail2 = prod(P_inv,2);
                tail = tail.*tail2;
                tmp =   mtimesx(1,cprodBefore(:,k,:),'t'); 
                head1 = bsxfun(@times, gradP(:,:,:,k), tmp);
                
                tmp = computeLeaveOneOut(P_inv_k_backup, timeDuration-1, batchSize);
                head2 = mtimesx(-gradP(:,1:end-1,:,k),tmp); 
                
                tmp =   mtimesx(1,P(2:timeDuration,k,:),'t'); 
                head3=bsxfun(@times, head2, tmp);
                head3 = [zeros(dim, 1, batchSize) head3];
                gradM(:,k1+1,:,k) = mtimesx(head1+head3,tail);
            else
                localcopy = cprodBefore;
                localcopy(:,k,:) = 1;
                head_common = prod(localcopy,2).*P(:,k1,:);  
                
                if k>k1
                    P_inv = 1-P;
                    P_inv_k_backup = P_inv(:,k,:);
                    P_inv(:,k1:end,:)=1;
                    head_common2 = prod(P_inv,2);
                    head_commonA = head_common.*head_common2;

                    tmp = computeLeaveOneOut(P_inv_k_backup, timeDuration-1, batchSize);
                    tail = mtimesx(-gradP(:,1:end-1,:,k),tmp); 
                    tail = [zeros(dim,1,batchSize) tail];
                    gradM(:,k1+1,:,k) = mtimesx(tail, head_commonA);                   
                             
                elseif k<k1
                    P_inv = 1-P;
                    P_inv_k_backup = P_inv(:,k,:);
                    P_inv(:,k1:end,:)=1;
                    P_inv(:,k,:)=1;
                    head_common2 = prod(P_inv,2);
                    head_commonB = head_common.*head_common2;  
                   
                    tmp = computeLeaveOneOut(P_inv_k_backup,  timeDuration, batchSize);
                    tail = mtimesx(-gradP(:,:,:,k),tmp);
                    gradM(:,k1+1,:,k) = mtimesx(tail, head_commonB);
                end
            end
        end
    end
    
    %% traverse all sample pairs
    for j=1:batchSize-1
        for i = j+1: batchSize    
            % do the subsampling for pairs        
            if labelset(j)~=labelset(i)
                if rand()>negativePairSampleRate 
                   continue;
                end
                %cntn = cntn+1;
            else
                if rand()>positivePairSampleRate 
                   continue;
                end
               %cntp = cntp+1;
            end

            hi_soft = h(:,:,j);
            hj_soft = h(:,:,i);

            % compute grad_H
            grad_H=zeros(dim, K);      
            for k = 1: K 
                grad_H(:,k) = gradM(:,:,i,k)*hi_soft + gradM(:,:,j,k)*hj_soft;
            end    
  
            H=0;
            hij = hi_soft'*hj_soft;
            % compute the training error of the pair (i,j)
            if labelset(j)~=labelset(i)
%                 if hij > 1-eps
%                    continue;
%                 end
%                 H = log(1-hij);  
%                 sum_gradW = sum_gradW + grad_H'/(1-hij); 

                H = hij;
                sum_gradW = sum_gradW + grad_H'; 
            else
%                 if hij< eps
%                     continue;
%                 end
%                 H = log(hij);               
%                 sum_gradW = sum_gradW - grad_H'/(hij);               
                
                H = 1-hij; 
                sum_gradW = sum_gradW - grad_H'; 
            end
            
            trainingError = trainingError + H;
                  
            cnt = cnt+1;  
            
        end %i  
    end % pair
    
    %% update projection
    gradWThis = learningRate*(sum_gradW/cnt+2*lambda1*LearnedProjection) + ...
                                                    lambda2*lastGradW_delta;
    LearnedProjection = LearnedProjection - gradWThis; 
    lastGradW_delta = gradWThis;
    currentNorm = norm(LearnedProjection);
    currentError = trainingError+lambda1*currentNorm;
    %LRate = [LRate; learningRate];   
%     if mod(epoch,50)==0
%         learningRate = learningRate*0.9;
%     end

    if currentError<previousError
        learningRate = learningRate*1.05;
    elseif currentError>previousError
        learningRate = learningRate*0.8;
    else
        learningRate = learningRate*1.0;
    end

%      normm = [normm; currentNorm];
      error = [error; currentError];
    epoch = epoch+1;
    previousError = currentError;
end 

elapse = toc(timee);

%drawing
% h1=figure(1);
% set(h1,'name','Projection Norm','NumberTitle','off');
% plot(normm); hold on;
% h2 = figure(2);
% set(h2, 'Name','Training objective','NumberTitle','off');
% plot(error); hold on;
% drawnow;

%elapse = toc(tstart);
 %fprintf('time elapse for one code = %f\n', elapse);
fprintf('error = %f, norm = %f, time = %.4f\n', trainingError, currentNorm, elapse);

end