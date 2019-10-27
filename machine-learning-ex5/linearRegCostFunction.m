function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

A = theta;
A(1)=0;
B= ones(length(A),1);
C= (lambda/(2*m))*(B'*(A.^2));

J = ((1/(2*m)) * ones(1,m) * ((((X*theta)-y).^2))) + C;

% Compute theta(0), the bias. Bias is first column of X full of ones, 
% hence directly taking ones
grad(1) = (1/m) * ((X*theta)-y)' * ones(m,1);

%Compute theta(1) onwards to end, taking second column of X to end, leaving the bias
grad(2:end,:) = ((1/m) * ((X*theta)-y)' * X(:,2:end))' + (lambda/m).*theta(2:end,:);






% =========================================================================

grad = grad(:);

end
