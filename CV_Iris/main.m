% main script
% Calvin Wang - 1/9/2016

% loading data 
fprintf('loading data') 

data = fopen('iris.data'); 
temp = textscan(data, '%f %f %f %f %s', 'Delimiter',','); 

% plotting data in 2D perspective -- sepal length (cm) vs sepal width (cm) 
fprintf('plotting sepal length (cm) vs sepal width (cm) data\n')

X = data(:, 1) ; 
y = data(:, 2) ; 
m = length(X) ;

plot(X, y) ; 