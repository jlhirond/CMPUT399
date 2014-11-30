function ev = cross_validate(x,y,ks,n)

training_feat_file='training_features.mat';
training_features=load(training_feat_file);
if nargin<4
    n=training_features.no_of_folds;
end
if nargin<3
    ks=training_features.ks;
end
if nargin<2
    y=training_features.y;
end
if nargin<1
    x=training_features.x;
end     

training_size=size(x, 2);
fold_size=ceil(training_size/n);
evall=zeros(n, numel(ks));
for k_id=1:numel(ks)
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
        yip = PredictPeopleCount(xi_(2:end,:), yi_, xi(2:end,:), k, 'mean');

        ei=mean(abs(yi-yip));
        evall(i, k_id)=ei;
    end
    fprintf('\tDone with k=%d\n', k);
    %fprintf('\n\tNumber of positive images: %d\n', positiveCount);
end
ev = mean(evall,1);
end