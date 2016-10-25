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

function HashCodes = ...
    ComputeFTAHashCodes(data, NormalizedProjection, numCodes, KK, ...
                       timeDuration, threshold, alpha)
                   
    product = data.feats*NormalizedProjection';
    
    %product1 = data.feats(:,1:24)*NormalizedProjection(:,1:24)';
    %sigmoid to produce the probability 
    P = 1./(1+exp(-alpha*product));
%     P(P<1e-4)=0;
%     P(P>1-1e-4)=1;
    HashCodes = ComputeHashBitMatrixProb(P, timeDuration, KK, numCodes); 
end
