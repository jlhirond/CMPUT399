function [W,b,accval] = BestWithSVM(Xtrain, ytrain, Xval, yval, C)
    epsilon = .000001;
    kerneloption= 1; % degree of polynomial kernel (1=linear)
    kernel='poly';   % polynomial kernel
    verbose = 0;
    Xtrain = Xtrain';
    Xval = Xval';
    [Xsup,yalpha,b,pos]=svmclass(Xtrain,ytrain,C,epsilon,kernel,kerneloption,verbose);
    [ypredtrain,acctrain,conftrain]=svmvalmod(Xtrain,ytrain,Xsup,yalpha,b,kernel,kerneloption);
    [ypredval,accval,confval]=svmvalmod(Xval,yval,Xsup,yalpha,b,kernel,kerneloption);
    W = (yalpha'*Xsup)';
    s=sprintf('C=%1.5f | Training accuracy: %1.3f; validation accuracy: %1.3f',C,acctrain,accval);
    fprintf([s '\n']);
end

