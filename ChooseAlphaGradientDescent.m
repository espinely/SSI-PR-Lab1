function [w] = ChooseAlphaGradientDescent(X, y, w)

NIter = 50;

alpha = 1:0.05:1.5;
alpha = log(alpha);
alpha= circshift(alpha,[-1 -1]); % return the vector to get a decrease order. 


curve1 = zeros(NIter + 1, 1);

curve1(1) = w(1);

N = size(X,1);


for j = 1:size(alpha,2)


    current_alpha = alpha(j);
    w_copy = w;
    disp(current_alpha)
        
    for k = 1:NIter
        Sum = 0;
        for i = 1:N
            Sum = Sum + (transpose(w_copy) * X(i) - y(i)) * X(i);
        end
        w_copy(1) = w_copy(1) - current_alpha * 1/N * Sum(1);

        curve1(k+1) = w_copy(1); % I study the change in w and not in J(w). 
    end
   
    
    figure;
    plot(curve1);
    str=sprintf('~~ChooseAlphaGradientDescent~~ Convergence of the GradientDescent for alpha = %d', current_alpha);
    title(str);
    xlabel('Number of iterations')
    ylabel('w')
    
    
    
    
    figure(50);  %Why 50? To insure that this figure number 50 is not used.
    hold on;
    plot(curve1);
    title('~~ChooseAlphaGradientDescent~~ Convergence of the GradientDescent for all the alpha');
    xlabel('Number of iterations')
    ylabel('w')
    
    

end

end








