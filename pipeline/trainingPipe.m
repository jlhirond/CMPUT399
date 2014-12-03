vlfeat_dir='vlfeat-0.9.19';
run(strcat(vlfeat_dir, '/toolbox/vl_setup'));
clear all;

SVM = 1;
DeCAF = 1;

% Tuning parameters
min_k=1;
max_k=10;
no_of_bins=32;
if (DeCAF == 1)
    no_of_bins = 4096;
end
no_of_folds=10;
combine_method='mean';
ks=min_k:max_k;
Call=[1000 100 10 1 .1 .01 .001 .0001 .00001];

% Mat files
training_feat_file='training_features.mat';
training_param='trained.mat';

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
y = [ones(posSize, 1);-ones(negSize, 1)];
indices = (1:posSize + negSize)';

% load positive decaf features
load(training_pos_file);
decaf_fv = transpose(decaf_fv);

% read all the images
tic
% bearFeature can be replaced with whatever feature we need to extract.

    %knn with histogram bear features
    if (DeCAF == 0)
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
    %knn with decaf features
    else
        % load positive decaf features
        load(training_pos_file);
        decaf_fv = transpose(decaf_fv);
        % read all the images
        tic
        for i = 1:posSize
             x(:,i) = decaf_fv(:, i);
             names(i,:) = {posNames(i).name};
        end
        load(training_neg_file);
        decaf_fv = transpose(decaf_fv);
        for i = 1:length(negNames1)
            x(:,posSize + i) = decaf_fv(:, i);
            names(posSize + i,:) = {negNames(i).name};
        end
        for i = 1:length(negNames2)
            x(:, posSize + length(negNames1) + i) = decaf_fv(:, length(negNames1) + i);
            names(posSize + length(negNames1) + i,:) = {negNames(length(negNames1)+i).name};
        end
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
    [ev, mParamW, mParamB] = cross_validate(X, Y, ks, no_of_folds, SVM);
    toc
    if (SVM == 0)
        [minev, minind] = min(ev);
        cross_validation_parameter = ks(minind(1));
        fprintf('Optimal k=%d\n', cross_validation_parameter);
        save(training_param, 'cross_validation_parameter');
    else
        [maxev, maxind] = max(ev);
        cross_validation_parameter = Call(maxind(1));
        fprintf('Optimal C=%d\n', cross_validation_parameter);
        Wbest = mParamW(maxind(1),:);
        Bbest = mParamB(maxind(1),:);
        save(training_param, 'Wbest', 'Bbest');
    end

% Extract localization features
fprintf('\nExtracting localization features from cropped images...\n');
posFilePattern = fullfile(pos_dir_name, 'cropped*.jpg');
croppedNames = dir(posFilePattern);
croppedCount = length(croppedNames);
cropped = cell(1, croppedCount);
croppedFeatures = cell(1, croppedCount);
% Load all of the images
for i=1:croppedCount;
    % read in the image
    im = imread(strcat(pos_dir_name, '/', croppedNames(i).name));
    % convert to grayscale
    im = rgb2gray(im);
    % resize to 24x24
    im = imresize(im, [24, 24]);
    % convert to single
    im = single(im);
    % add to cropped cell array
    cropped{i} = im;
end
% Create an array of images
for i=1:croppedCount
    % compute vl_hog features for each image and place in feature matrix
    croppedFeatures{i} = vl_hog(cropped{i}, 8);
end
% save the results
save('localization_features.mat', 'croppedFeatures');
fprintf('Done.\n');
