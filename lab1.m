%% FIRST PROBLEM:

clc;clear all;close all;
%Import data:
filename = 'lab1data1.txt';
%and split information separed by commas:
delimiterIn = ',';
A = importdata(filename,delimiterIn);
plotData(A(:,1),A(:,2));
%Perform linear regression with the data:
w = linearReg(A(:,1),A(:,2))
%Calculate the profits for a city with 35,000 inhabitants:
y1 = w' * [1;3.5]
%Calculate the profits for a city with 70,000 inhabitants:
y2 = w' * [1;7]
%From the results, we can see that the profits are of $2798 and $44555.

%% SECOND PROBLEM:

clc;clear all;close all;
%Import data with two features:
filename = 'lab1data2.txt';
delimiterIn = ',';
A = importdata(filename,delimiterIn);
%Perform linear regression without feature normalization:
w = linearReg(A(:,1:2),A(:,3))



%Perform feature normalization:
[Xn,mu,sigma] = featureNormalize(A(:,1:2))
%Perform linear regression with the normalized data:
w = linearReg(Xn,A(:,3))
%Calculate the price of a house of size 1650 square feet and 3 bedrooms:
houseSize = 1650;
houseBedrooms = 3;
houseSize = (houseSize - mu(1))/sigma(1);
houseBedrooms = (houseBedrooms - mu(2))/sigma(2);
y = w' * [1;houseSize;houseBedrooms]
%After computing the price of the house, the obtained value is $293080.

%% THIRD PROBLEM:
clc;clear all;close all;
filename = 'lab1data2.txt';
delimiterIn = ',';
A = importdata(filename,delimiterIn);
%Perform feature normalization:
[Xn,mu,sigma] = featureNormalize(A(:,1:2));
%ALGORITHM TO CHOOSE ALPHA (Learning Rate):
%List of alpha's:
alpha=[0.3 0.2 0.03 0.01 0.003 0.001];
%For every alpha:
for ialpha=1:length(alpha)
    %Select number of iterations:
    NIter=50;
    %Compute the gradient descent:
    [w,J] = gradientDescent(Xn,A(:,3),alpha(ialpha),NIter);
    %Calculate the price of house for 1650 square feet and 3 bedrooms:
    houseSize = 1650;
    houseBedrooms = 3;
    %Compute the normalized features:
    houseSize = (houseSize - mu(1))/sigma(1);
    houseBedrooms = (houseBedrooms - mu(2))/sigma(2);
    %Compute the price of the house:
    y(ialpha) = w(:,NIter)' * [1;houseSize;houseBedrooms];
    figure;
    %Plot the evolution of the cost function
    plot([1:NIter],J);
end
%After seeing the cost functions for the different learning rates, I can determine
%that the best alpha is 0.3, as it's the one which converges the fastest
%to the right value.