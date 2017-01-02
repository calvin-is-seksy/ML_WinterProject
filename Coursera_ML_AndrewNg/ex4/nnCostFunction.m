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

a1 = [ones(m,1) X];		% 5000x401
				% Theta1 25x401
z2 = a1*transpose(Theta1);	% 5000x25
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];		% 5000x26
				% Theta2 10x26
z3 = a2*transpose(Theta2);	% 5000x10
h = sigmoid(z3);		% h 5000x10

% convert y value from 1:num_labels to num_labelsx1 matrix with only 1 ones and other zeros
yy = zeros(m, num_labels);
for kk = 1:m,
  %yy(kk,mod(y(kk),num_labels)+1) = 1;
  yy(kk,y(kk)) = 1;
end
				% yy 5000x10

J = 1/m*sum(sum(-yy.*log(h)-(1-yy).*log(1-h))) ...
+lambda/2/m*sum(sum(Theta1(:,2:end).^2)) ...
+lambda/2/m*sum(sum(Theta2(:,2:end).^2));

% backpropagation

big_delta1 = zeros(size(Theta1));	% 25x401
big_delta2 = zeros(size(Theta2));	% 10x26

for t = 1:m,
  aa3 = transpose(h(t,:));		% 10x1
  delta3 = aa3 - transpose(yy(t,:));	% 10x1
  delta2 = transpose(Theta2(:,2:end))*delta3.*transpose(sigmoidGradient(z2(t,:))); % 25x1
  big_delta1 = big_delta1 + delta2 * a1(t,:);	% 25x401
  big_delta2 = big_delta2 + delta3 * a2(t,:);	% 10x26 
end

Theta1_temp = Theta1;
Theta1_temp(:,1) = 0;
Theta2_temp = Theta2;
Theta2_temp(:,1) = 0;
Theta1_grad = 1/m*big_delta1 + lambda/m*Theta1_temp;
Theta2_grad = 1/m*big_delta2 + lambda/m*Theta2_temp;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
