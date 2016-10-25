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

function tmp = computeLeaveOneOut(P_inv_k_backup,  timeDuration, batchSize)

    tmp = zeros(timeDuration, timeDuration, batchSize);
    for t = 1:timeDuration
        for tt = 1:t
            P_inv_k = P_inv_k_backup;
            P_inv_k(tt,:,:) = 1;
            tmp(tt,t,:) = prod(P_inv_k(1:t,:,:));
        end     
    end
end