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

% script for boosting
clc
addpath '../mtimesx';
seed = 'default'

CodeNum =128;
K=3;
dim = 6;
learningRate = 1;
lambda1 = 0;
alpha1 = 5;
MaxEpoch = 30;
threshold = [1.3 1.2 1.3 1.6 1.6];
for sub = 1:5
    fprintf('Training sub %d, threshold =  %.4f, K=%d, CodeLen = %d\n', ...
                                            sub, threshold(sub), K, CodeNum); 
    LearnScriptNoBoosting(seed, dim, CodeNum, K, sub, ...
                 learningRate, lambda1, alpha1, MaxEpoch, threshold(sub));
end