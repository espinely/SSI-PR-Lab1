function w = linearReg(X, y)
    %Initialize phi matrix with first column equal to one
    phi=[ones(size(X,1),1),X];
    for i=1:size(X,1)
        phi(i,1)=1;
        %Add the different features as columns
        for j=1:size(X,2)
            phi(i,j+1)=X(i,j);
        end
    end
    %Compute the weight matrix
    w = inv(phi' * phi) * phi' * y;