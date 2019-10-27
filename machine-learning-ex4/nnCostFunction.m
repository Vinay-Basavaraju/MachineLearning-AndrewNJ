function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Comments added by Vinay
%Theta1 = 25*401, Theta2 = 10*26 for this example
%input_layer_size = 400, hidden_layer_size = 25, num_labels = 10, m = 5000 for this example

% Add ones to the X data matrix
X = [ones(m, 1) X];

% To store the sum of cost for all the clases in each example 
temp_cost = 0;

for i = 1:m
  
  a1 = X(i,:);
  
  a2 = sigmoid(Theta1*a1');
  
  a2 = [1;a2];
  
  a3 = sigmoid(Theta2*a2);
  
  %Form matrices of zeros and ones where 1 indicates the label for the current example
  y_temp = [1:num_labels]';
    
  y_temp = (y(i) == y_temp);
    
  %Sum of cost for all classes
  temp_cost = temp_cost + (ones(1,num_labels) * ( (-y_temp.*log(a3)) - ((1-y_temp).*log(1 - a3)) ) );
    
endfor

J = (1/m) * temp_cost;


% Adding regularization to cost
% Adding zeros to first column of Theta1 & Theta2 since Bias should not be regularized
Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;

Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;

%All the elements in Theta should be squared and summed up, hence first 
%all rows in the Theta1 will be summed up individually then all columns will be summed up
B = ones(input_layer_size+1,1);
C = (Theta1_temp.^2) * B;
B = ones(hidden_layer_size,1);
D = B' * C;

B = ones(hidden_layer_size+1,1);
C = (Theta2_temp.^2) * B;
B = ones(num_labels,1);
E = B' * C;

Reg_Cost = (lambda/(2*m))*(D+E);

J = J + Reg_Cost;


%%Backpropagation
%DeltaAccumulate_Layer1 = 25* 401
DeltaAccumulate_Layer1 = zeros(hidden_layer_size,input_layer_size+1);

%DeltaAccumulate_Layer2 = 10* 26
DeltaAccumulate_Layer2 = zeros(num_labels,hidden_layer_size+1);

for i = 1:m
  %a1=1*401
  a1 = X(i,:);
  
  a2 = sigmoid(Theta1*a1');
  
  %a2=26*1
  a2 = [1;a2];
  
  %a3=10*1
  a3 = sigmoid(Theta2*a2);
  
  y_temp = [1:num_labels]';
    
  y_temp = (y(i) == y_temp);
  
  %error_term_3 = 10*1
  error_term_3 = a3 - y_temp;
  
  %Theta2_temp = 26*10
  %Theta2_temp = Theta2';
  
  %Removing the first row of bias
  %Theta2_temp(1,:) = [];
  
  
  a2_SG=sigmoidGradient(Theta1*a1');
  %a2_SG=26*1
  a2_SG = [1;a2_SG];
  
  
  %error_term_2 = 26*1
  error_term_2 = (Theta2'*error_term_3).*a2_SG;
  

  DeltaAccumulate_Layer1 = DeltaAccumulate_Layer1 + (error_term_2(2:end)*a1);
  
  DeltaAccumulate_Layer2 = DeltaAccumulate_Layer2 + (error_term_3*a2');
  
endfor

Theta1_grad = (1/m).* DeltaAccumulate_Layer1;
Theta2_grad = (1/m).* DeltaAccumulate_Layer2;

%Theta1_grad = [ones(hidden_layer_size, 1) DeltaAccumulate_Layer1];
%Theta2_grad = [ones(num_labels, 1) DeltaAccumulate_Layer2];

%Store First column
Theta1_grad_FC = Theta1_grad(:,1);
Theta2_grad_FC = Theta2_grad(:,1);;

%Regularization for Gradients
%Skip the first column which is the bias
Theta1_grad = Theta1_grad(:,2:end) + (lambda/m).*Theta1(:,2:end);
Theta2_grad = Theta2_grad(:,2:end) + (lambda/m).*Theta2(:,2:end);

Theta1_grad = [Theta1_grad_FC Theta1_grad];
Theta2_grad = [Theta2_grad_FC Theta2_grad];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
