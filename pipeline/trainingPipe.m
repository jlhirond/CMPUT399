function trainingPipe(classifier, feature)
vlfeat_dir='C:/Users/Valerie/Documents/MATLAB/vlfeat-0.9.19';
run(strcat(vlfeat_dir, '/toolbox/vl_setup'));
addpath(genpath('cvml2013-practical-face-detection'));
h=waitbar(0,'Extracting localization features from positive images...');
waitObject = onCleanup(@() delete(h));

if nargin < 2
    feature = 'DeCAF';
end
if nargin < 1
    classifier = 'SVM';
end

% Tuning parameters
min_k=1;
max_k=10;
no_of_bins=32;
if (strcmp(feature,'DeCAF'))
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

% local features
% Extract localization features
tic
fprintf('\nExtracting localization features from cropped images...\n');
posFilePattern3 = fullfile(pos_dir_name, 'cropped*.jpg');
croppedNames = dir(posFilePattern3);
croppedPosCount = length(croppedNames);
cropped = cell(1, croppedPosCount);
croppedPosFeatures = cell(1, croppedPosCount);
croppedDimensions = zeros(croppedPosCount, 4);
% Load all of the images
for i=1:croppedPosCount;
    
    % get the image name
    cropName = char(croppedNames(i).name);
    % split on the . to get rid of the end
    cropName = strsplit(cropName, '.');
    % split on all of the _
    cropName = strsplit(cropName{1}, '_');
    % read in the image
    im = imread(strcat(pos_dir_name, '/', posNames(i).name));
    % get the larger of the height and width variable
    cropSize = max(str2num(cropName{5}), str2num(cropName{6}));
    % crop the image based on the extracted file location
    im = imcrop(im,[str2num(cropName{3}), str2num(cropName{4}), cropSize, cropSize]);
    % save all location values for later use
    for j=1:4;
        croppedDimensions(i, j) = str2num(cropName{j+2});
    end
    % convert to grayscale
    im = rgb2gray(im);
    % resize to 24x24
    im = imresize(im, [24, 24]);
    % convert to single
    im = im2single(im);
    % add to cropped cell array
    cropped{i} = im;
    waitbar(i/croppedPosCount);
end

% get cropped neg images
%croppedNegatives = cell(576, negSize);
croppedNegatives = cell(1, negSize);
waitbar(0,h,'Extracting localization features from negative images...');
for i=1:negSize
    rand_loc = randperm(croppedPosCount,1);
    dimensions = croppedDimensions(rand_loc,3);
    Xcorner = croppedDimensions(rand_loc,1);
    Ycorner= croppedDimensions(rand_loc,2);
    
    % Read from the correct folder
    if i > length(negNames2)
        dir_name = 'NegativeImages1';
    else
        dir_name = 'NegativeImages2';
    end
    
    imNeg = imread(strcat(dir_name, '/', negNames(i).name));
    croppedNeg = imNeg(Xcorner:Xcorner+dimensions, Ycorner:Ycorner+dimensions);
    % convert to grayscale
    %croppedNeg = rgb2gray(croppedNeg);
    % resize to 24x24
    croppedNeg = imresize(croppedNeg, [24, 24]);
    % convert to single
    croppedNeg = im2single(croppedNeg);
    % add to cropped cell array
    croppedNegatives{i} = croppedNeg;
    waitbar(i/negSize);
end

% Create an array of images
for i=1:croppedPosCount
    % compute vl_hog features for each positive image and place in feature matrix
    feat = vl_hog(cropped{i}, 24);
    PosFeat{i} = feat(:);
end
for k=1:negSize
    % compute vl_hog features for each negative image and place in feature matrix
    feat = vl_hog(croppedNegatives{k}, 24);
    NegFeat{k} = feat(:);
end
croppedPosFeatures = (cell2mat(PosFeat));
croppedNegFeatures = (cell2mat(NegFeat));

croppedPosFeatures = croppedPosFeatures';
croppedNegFeatures = croppedNegFeatures';

% save the results
save('localization_features.mat', 'croppedPosFeatures', 'croppedNegFeatures');
save('cropped_images.mat', 'cropped', 'croppedNegatives');
toc
% read all the images
disp('Loading names and counts of training images...');
waitbar(0,h,'Reading images...');
tic
% bearFeature can be replaced with whatever feature we need to extract.

%knn with histogram bear features
if (strcmp(feature, 'hist'))
    for i = 1:posSize
        img = imread(strcat(pos_dir_name, '/', posNames(i).name));
        x(:,i) = bearFeature(img);
        waitbar(i/(posSize+negSize));
    end
    for i = 1:length(negNames2)
        img = imread(strcat(neg_dir_name2, '/', negNames(i).name));
        x(:,posSize + i) = bearFeature(img);
        waitbar((posSize+i)/(posSize+negSize));
    end
    for i = 1:length(negNames1)
        img = imread(strcat(neg_dir_name1, '/', negNames(length(negNames2)+i).name));
        x(:, posSize + length(negNames2) + i) = bearFeature(img);
        waitbar((posSize+length(negNames2)+i)/(posSize+negSize));
    end
    %with decaf features
else
    % load positive decaf features
    load(training_pos_file);
    testCaf(1,:) = decaf_fv(18,:);
    testCaf(2,:) = decaf_fv(43,:);
    decaf_fv = removerows(decaf_fv,'ind',[18 43]); %******************************************************
    decaf_fv = transpose(decaf_fv);
    % read all the images
    tic
    for i = 1:posSize
        x(:,i) = decaf_fv(:, i);
        names(i,:) = {posNames(i).name};
        waitbar(i/(posSize+negSize));
    end
    load(training_neg_file);
    testCaf(3,:) = decaf_fv(19,:);
    testCaf(4,:) = decaf_fv(1093,:);
    testCaf(5,:) = decaf_fv(1238,:);
    testCaf(6,:) = decaf_fv(1958,:);
    decaf_fv = removerows(decaf_fv,'ind',[19 1093 1238 1958]); %**********************************************
    decaf_fv = transpose(decaf_fv);
    for i = 1:length(negNames1)
        x(:,posSize + i) = decaf_fv(:, i);
        names(posSize + i,:) = {negNames(i).name};
        waitbar((posSize+i)/(posSize+negSize));
    end
    for i = 1:length(negNames2)
        x(:, posSize + length(negNames1) + i) = decaf_fv(:, length(negNames1) + i);
        names(posSize + length(negNames1) + i,:) = {negNames(length(negNames1)+i).name};
        waitbar((posSize+length(negNames2)+i)/(posSize+negSize));
    end
    decaf_fv = testCaf;
    save('test.mat','decaf_fv');
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
[ev, param] = cross_validate(X, Y, ks, no_of_folds, classifier, h);
toc
if (strcmp(classifier,'KNN'))
    [minev, minind] = min(ev);
    cross_validation_parameter = ks(minind(1));
    fprintf('Optimal k=%d\n', cross_validation_parameter);
    k = cross_validation_parameter;
    save(training_param, 'k');
else
    [maxev, maxind] = max(ev);
    cross_validation_parameter = Call(maxind(1));
    fprintf('Optimal C=%d\n', cross_validation_parameter);
    Wbest =param(maxind(1),:).W;
    Bbest = param(maxind(1),:).B;
    save(training_param, 'Wbest', 'Bbest');
end
fprintf('Done.\n');
end