function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%r = size(z,1);
%c = size(z,2);

  %for i = 1:r
   % for j = 1:c
    %  g(i,j) = 1/(1+exp(-z(i,j)));
    % endfor
   %endfor
  g = 1./(1.+exp(-z));
% =============================================================

end
