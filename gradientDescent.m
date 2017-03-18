function [w,J] = gradientDescent(X,y,alpha,NIter)
    %Initialize weight matrix
    w=zeros(size(X,2)+1,NIter);
    %Add a column of ones to the features
    X=[ones(size(X,1),1),X];
    %For every iteration:
    for i=1:NIter
        suma=0;
        %For every feature:
        for j=1:size(X,2)
            %Compute the weight update:
            if i>1
            	suma= sum((X*w(:,i-1) - y).*X(:,j));
            else
                suma= sum((X*w(:,i) - y).*X(:,j));
            end
            deriv=suma/size(X,1);
            u(j) = alpha*deriv;
        end
        %Correct the weights according to the update:
        if i>1
            w(:,i) = w(:,i-1) - u';
        else
            w(:,i) = - u;
        end
        %Copmute the cost function:
        suma=0;
        for j=1:size(X,1)
            suma=suma + (X(j,:)*w(:,i) - y(j,:))^2;
        end
        J(i)=suma/(2*size(X,1));
        
    end