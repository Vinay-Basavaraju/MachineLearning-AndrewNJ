function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


A = theta;
A(1)=0;
B= ones(length(A),1);
C= (lambda/(2*m))*(B'*(A.^2));

J = ((1/m) * ones(1,m) * ( (-y.*log(sigmoid(X*theta))) - ((1-y).*log(1 - sigmoid(X*theta))) )) + C;

A = A.*(lambda/m);

grad = (( (1/m) * ((sigmoid(X*theta) - y)' * X) )').+ A;


% =============================================================

end
