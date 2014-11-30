vlfeat_dir='vlfeat-0.9.19';
run(strcat(vlfeat_dir, '/toolbox/vl_setup'));

clear all;

% Tuning parameters
min_k=1;
max_k=10;
no_of_bins=32;
no_of_folds=10;
combine_method='mean';
ks=min_k:max_k;

% Mat files
training_feat_file='training_features.mat';
training_param='trained.mat'

training_pos_file='01_pos.mat';
training_neg_file='02_neg.mat';

% Load the images
disp('Loading names and counts of training images...');
pos_dir_name = 'PositiveImages';
neg_dir_name1 = 'NegativeImages1';
neg_dir_name2 = 'NegativeImages2';

posFilePattern = fullfile(pos_dir_name, 'bear*.jpg');
negFilePattern1 = fullfile(neg_dir_name1, 'morantBridgeSite*.jpg');
negFilePattern2 = fullfile(neg_dir_name2, 'morantBridgeSite*.jpg');

% Get the dir struct of all the images
posNames = dir(posFilePattern);
negNames1 = dir(negFilePattern1);
negNames2 = dir(negFilePattern2);
negNames = [negNames2;negNames1];

posSize = length(posNames);
negSize = length(negNames);
% The feature and response vectors, respectively.

x = zeros(no_of_bins, posSize + negSize);
y = [ones(posSize, 1);zeros(negSize, 1)];
indices = (1:posSize + negSize)';

% read all the images
tic
% bearFeature can be replaced with whatever feature we need to extract.
for i = 1:posSize
    img = imread(strcat(pos_dir_name, '/', posNames(i).name));
     x(:,i) = bearFeature(img, 'hist');
end
for i = 1:length(negNames2)
    img = imread(strcat(neg_dir_name2, '/', negNames(i).name));
    x(:,posSize + i) = bearFeature(img, 'hist');
end
for i = 1:length(negNames1)
    img = imread(strcat(neg_dir_name1, '/', negNames(length(negNames2)+i).name));
    x(:, posSize + length(negNames2) + i) = bearFeature(img, 'hist');
end

% randomly rearrange images so that all the positive images are not contained within one fold
orderedArray = horzcat(indices,x', y);
shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);
X = shuffledArray(:,1:end-1)'; %(:,2:end-1)'
Y = shuffledArray(:,end);

% save everything in a .mat file for ease of use, testing, etc.
save(training_feat_file, 'shuffledArray','X', 'Y', 'ks', 'no_of_folds');
toc

% perform cross validation in order to find the optimal value for our
% tuning parameter(s).
disp('Performing cross validation...');
tic
ev = cross_validate(X, Y, ks, no_of_folds);
toc

[minev, minind] = min(ev);
cross_validation_parameter = ks(minind(1));
fprintf('Optimal k=%d\n', cross_validation_parameter);
save(training_param, 'cross_validation_parameter');
