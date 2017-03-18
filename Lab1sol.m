
% ----------------------------------------------------
% II Linear regression with one variable 
% ----------------------------------------------------


data1 = importdata('lab1data1.txt');
population = data1(:,1);
profit = data1(:,2);

plotData(population, profit);

w_Data1 = LinearReg(population, profit);


%If population = 35000 or 70000, profit?

profit1 = (w_Data1(1) + w_Data1(2) * 3.5000) * 10000; % * 10000 because of the scale. 
profit2 = (w_Data1(1) + w_Data1(2) * 7.0000) * 10000;





% ----------------------------------------------------
% III Linear regression with multiple variables
% ----------------------------------------------------


data2   =  importdata('lab1data2.txt');
x_Data2 = [data2(:,1), data2(:,2)];
y_Data2 =  data2(:,3);
w_Data2 =  LinearReg(x_Data2, y_Data2);
price   =  w_Data2(1) + w_Data2(2) * 1650 + w_Data2(3) * 3;

 
   
[x_Data2_norm, mu, sigma] = featureNormalize(x_Data2);
w_Data2_norm = LinearReg(x_Data2_norm, y_Data2);


size_norm = (1650 - mu(1)) / sigma(1);
bedrooms_norm = (3  - mu(2)) / sigma(2);
price_norm = w_Data2_norm(1) + w_Data2_norm(2) * size_norm + w_Data2_norm(3) * bedrooms_norm;

% size_norm = size
% bedrooms_norm = bedrooms
% price_norm = price

% With the normalization, the results are the same but the calculations are less heavy. 


%%

% ----------------------------------------------------
% IV Gradient descent
% ----------------------------------------------------

data2   =  importdata('lab1data2.txt');
x_Data2 = [data2(:,1), data2(:,2)];
y_Data2 =  data2(:,3);
[x_Data2_norm, mu, sigma] = featureNormalize(x_Data2);

 w_GradDescent = zeros(size(x_Data2,2),2);
 NIter = 50;
 alpha = 0.17;
 w_GradDescent = GradientDescentSol(x_Data2_norm, y_Data2, w_GradDescent, alpha, NIter);

 size_GradDescent = (1650 - mu(1)) / sigma(1);
 bedrooms_GradDescent = (3  - mu(2)) / sigma(2);
 price_GradDescent = w_GradDescent(1) + w_GradDescent(2) * size_norm + w_GradDescent(3) * bedrooms_norm;


 w_GradDescent = ChooseAlphaGradientDescent(x_Data2_norm, y_Data2, w_GradDescent);
 % We can take alpha = 1.397619*exp(-1).