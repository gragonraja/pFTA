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

%compute the probability of the leading pattern among K patterns

function leadingProb = ComputeLeadingProbMatrix(P, T, numCodes, K)
    num = size(P, 3);
    P_inv = 1-P;
    P_inf = prod(P_inv);
    P_inf = reshape(P_inf, K, numCodes*num);
    h0= prod(P_inf);
    prod_prefix = cumprod(P_inv);
    P3d = reshape(P, T, K, []);
    S3d = reshape(prod_prefix, T, K, []);
    P3dInv = reshape(P_inv, T, K, []);
    hh = zeros(K, numCodes*num);
    prod_K = prod(S3d, 2);
    
    for k=1:K 
        
         tmp = [ones(1,1,numCodes*num); prod_K(1:end-1,:,:)];
         for k1 =1:K
             if k1<k % include the tie and encode the smallest index
                 tmp = tmp.*P3dInv(:,k1,:);
             end
         end
        
        tmp2 =  mtimesx(P3d(:,k,:),'T', tmp);
        hh(k,:) = reshape(tmp2,1,[]);
    end
   
    leadingProb = [h0; hh];
    
%     leadingProb(leadingProb<1e-4)=0;
%     leadingProb(leadingProb>(1-1e-4)) = 1;
    
 end