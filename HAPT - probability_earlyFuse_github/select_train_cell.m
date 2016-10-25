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

function [trainCell, labelset, TT] = select_train_cell(data, batchSize, ...
    subsetIndex, timeDuration, dim)

    labelset=zeros(batchSize,1);
    %trainCell=cell(batchSize,1);
    trainCell = zeros(dim, timeDuration, batchSize);
    TT=timeDuration*ones(batchSize,1);

    for k=1:batchSize
        offset = (subsetIndex(k)-1)*timeDuration;
        trainCell(:,:,k) = data.feats(offset + 1: offset + timeDuration,:)';
        labelset(k) = data.class(subsetIndex(k));
    end
end