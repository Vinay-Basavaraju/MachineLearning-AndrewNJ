function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);

b = ones(size(X,2),1);

XCentrDiff = zeros(m,K);

%Subtract each input with the centroid. The ith column of XCentrDiff will have 
%the difference of the input with the i th centroid
for i= 1:K
  
  XTemp = X - centroids(i,:);  
  
  XTemp = XTemp.^2;
  
  %Add all the elements or columns of each input (x1 + x2)
  XCentrDiff(:,i) = XTemp * b;
  
  
endfor

for i= 1:m
  
%index of the smallest difference with the centroid. Index number indicates the 
%centroid. Centroid number is assigned to idx(i). Where i is the input row number 
  idx(i) = find(XCentrDiff(i,:) == min(XCentrDiff(i,:)),1);
  
endfor





% =============================================================

end

