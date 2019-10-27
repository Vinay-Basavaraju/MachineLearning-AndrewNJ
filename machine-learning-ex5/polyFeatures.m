function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
  %Number of training examples
  m = numel(X);
  for i= 1:m
    
    %Hold the i th row of training example
    x_temp = X(i,:); 
    
    for j = 2:p
      
      %Keep adding additional column for each degree of polynomial on 
      %the first column value only
      x_temp = [x_temp x_temp(1).^j];      
            
    endfor    
    
    %Store the complete i th row of polynomial
    X_poly(i, :) = x_temp;
    
  endfor




% =========================================================================

end
