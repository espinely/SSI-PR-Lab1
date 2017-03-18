function [w] = GradientDescentSol(X, y, w, alpha, NIter)

curve1 = zeros(NIter + 1, 1);


curve1(1) = w(1);

N = size(X,1);

for k = 1:NIter
    Sum = 0;
    for i = 1:N
        Sum = Sum + (transpose(w) * X(i) - y(i)) * X(i);
    end
    w(1) = w(1) - alpha * 1/N * Sum(1);
  
    curve1(k+1) = w(1);
    
end

figure;

plot(curve1);
title('~~GradientDescent~~ Convergence of the GradientDescent')
xlabel('Number of iterations')
ylabel('w')







