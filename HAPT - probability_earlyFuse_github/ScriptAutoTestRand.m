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

% script for batch running
clear all;
addpath '../mtimesx'
seed = 'default';
CodeNum =1
K=3;
dim = 6;
alpha=5;

knn = 15;

threshold = 1.4;
accuracy=[];
for sub = 1:5
    t=tic;
    tmp = fusionRand(seed, K, dim, CodeNum, sub, threshold, knn, alpha);
    elapse = toc(t);
    fprintf('sub: %d, accuracy = %.4f, elapse = %.4f\n', sub, max(tmp), elapse);
    accuracy = [accuracy tmp];

end
meanAvg = mean(accuracy,2);
[maxAvg,index]= max(meanAvg);
fprintf('threshold = %.4f, mean accuracy = %.4f @ knn = %d\n', threshold, maxAvg, index);

