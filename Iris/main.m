% main script
% Calvin Wang - 1/9/2016

% loading data 
fprintf('loading data\n') 

data = fopen('iris.data'); 
temp = textscan(data, '%f %f %f %f %s', 'Delimiter',','); 
sepal_L = temp{1} ; 
sepal_W = temp{2} ; 
petal_L = temp{3} ; 
petal_W = temp{4} ; 
class = temp{5} ; 

% plotting data in 2D perspective -- sepal length (cm) vs sepal width (cm) 
fprintf('plotting sepal length (cm) vs sepal width (cm) data\n')

m = length(sepal_L) ;

figure(1) ; 
plot(sepal_L, sepal_W, 'bx') 
xlabel('Sepal Length (cm)') ;
ylabel('Sepal Width (cm)') ; 

% plotting data in 2D perspective -- petal length (cm) vs petal width (cm) 
fprintf('plotting petal length (cm) vs petal width (cm) data\n')

figure(2) ; 
plot(petal_L, petal_W, 'rx') ; 
xlabel('Petal Length (cm)') ; 
ylabel('Petal Width(cm)') ; 

% gradient descent to fit a linear regression 
fprintf('Now let us run some gradient descent:\n') 
X = [ones(m, 1), petal_L]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 15000;
alpha = 0.01;

% compute and display initial cost
computeCost(X, petal_W, theta)

% run gradient descent
theta = gradientDescent(X, petal_W, theta, alpha, iterations)

% Plot the linear fit
hold on; 
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off; 

% predict values for petal lengths 3 cm and 6 cm 
predict1 = [1, 3] * theta; 
fprintf('For a petal length of 3 cm, we predict a petal width of %f cm.\n',... 
    predict1); 

predict2 = [1, 6] * theta; 
fprintf('For a petal length of 6 cm, we predict a petal width of %f cm.\n',... 
    predict2); 

% visualizing J(theta_0, theta_1) 
fprintf('Visualizing J(theta_0, theta_1) ...\n') 

% Grid over which I will calculate J 
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, sepal_W, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

