function [ev, param] = cross_validate(x,y,ks,n,classifier,h)
training_feat_file='training_features.mat';
training_features=load(training_feat_file);
if nargin<6
    h = waitbar(0,'Performing cross validation...');
end
if nargin<5
    classifier = 'SVM';
end
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
Call=[.001 .0001 .00001];

if (strcmp(classifier,'KNN'))
    loops = numel(ks);
else
    loops = length(Call);
end
total = loops * n;

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
        if (strcmp(classifier, 'KNN'))
            yip = PredictPeopleCount(xi_(2:end,:), yi_, xi(2:end,:), k, 'mean');
            ei=mean(abs(yi-yip));
            evall(i, k_id)=ei;
            paramK(i,:)=k;
        else
            %do SVM
            C = Call(k_id);
            [w, b, accval] = BestWithSVM(xi_(2:end,:), yi_, xi(2:end,:), yi, C);
            evall(i, k_id) = accval;
            paramW(i,:) = w;
            paramB(i,:) = b;
        end
        progress = n*k_id + i - n;
        waitbar(progress/total,h,'Performing cross validation...');
    end
    fprintf('\tDone with k=%d\n', k);
    if (strcmp(classifier,'SVM'))
        param(k_id,:).W = mean(paramW,1);
        param(k_id,:).B = mean(paramB,1);
    else
        param(k_id,:).K = mean(paramK,1);
    end
end
ev = mean(evall,1);
end
