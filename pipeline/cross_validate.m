function [ev, mParamW, mParamB] = cross_validate(x,y,ks,n,SVM)
training_feat_file='training_features.mat';
training_features=load(training_feat_file);
if nargin<4
    n=training_features.no_of_folds;
end
if nargin<3
    ks=training_features.ks;
end
if nargin<2
    y=training_features.Y;
end
if nargin<1
    x=training_features.X;
end     

training_size=size(x, 2);
fold_size=ceil(training_size/n);
evall=zeros(n, numel(ks));
%for SVM
Call=[1000 100 10 1 .1 .01 .001 .0001 .00001];
if SVM == 0
    loops = nume1(ks);
else
    loops = 9; % yup
end
for k_id=1:loops
    k=ks(k_id);
    for i = 1:n
        fold_start_idx=(i-1)*fold_size + 1;
        if i ~= n
            fold_end_idx=i*fold_size;
        else
            fold_end_idx=size(x,2);
        end
        % get validation data
        xi=x(:, fold_start_idx:fold_end_idx);
        yi=y(fold_start_idx:fold_end_idx);
        % get training data
        other_folds_idx=[1:fold_start_idx-1 fold_end_idx+1:training_size];
        xi_=x(:, other_folds_idx);
        yi_=y(other_folds_idx);
        % this can be svm, sift, etc.
        %yip = PredictBear(xi_, yi_, xi, k);
        if SVM == 0
            yip = PredictPeopleCount(xi_(2:end,:), yi_, xi(2:end,:), k, 'mean');
            ei=mean(abs(yi-yip));
            evall(i, k_id)=ei;
        else
            %do SVM
            C = Call(k_id);
            [w, b, accval] = BestWithSVM(xi_(2:end,:), yi_, xi(2:end,:), yi, C);
            evall(i, k_id) = accval;
            paramW(i,:) = w;
            paramB(i,:) = b;
        end
    end
    fprintf('\tDone with k=%d\n', k);
    mParamW(k_id,:) = mean(paramW,1);
    mParamB(k_id,:) = mean(paramB,1);
    %fprintf('\n\tNumber of positive images: %d\n', positiveCount);
end
ev = mean(evall,1);
end