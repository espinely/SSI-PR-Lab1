function [Xn,mu,sigma] = featureNormalize(X)
    %For every sample
    for i=1:size(X,1)
        %And every feature
        for j=1:size(X,2)
            %Compute the mean of all the samples of the feature
            xMean=mean(X(:,j));
            %And also the standard deviation
            xStd=std(X(:,j));
            %Compute the normalized sample
            Xn(i,j) = (X(i,j) - xMean) / xStd;
            mu(j)=xMean;
            sigma(j)=xStd;
        end
    end