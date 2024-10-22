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

function NormProjection = GaussianRandomGenerator(dim, num)
mu = zeros(1,dim);
sigma = eye(dim);

%rng(seed)  % For reproducibility
NormProjection = mvnrnd(mu,sigma,num);
NormProjection = normr(NormProjection);
% for i=1:num
% NormProjection(i,:) = randomProjection(i,:)/norm(randomProjection(i,:));%*8;
% end