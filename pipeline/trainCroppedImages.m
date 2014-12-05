function trainCroppedImages(classifier)
if nargin<1
    classifier = 'SVM';
end
ks=1:10;
n = 10;
load('localization_features.mat');
load('cropped_images.mat', 'cropped');
load('cropped_images.mat', 'croppedNegatives');
Ypos = ones(size(croppedPosFeatures,1),1);
Yneg = -ones(size(croppedNegFeatures,1),1);

croppedFeatures = [croppedPosFeatures; croppedNegFeatures];

for i =1:length(cropped)
    img = double(cropped{:,i});
    croppedPosImages(:,:,i) = img;
end
croppedNegatives = croppedNegatives(:)';
for i =1:length(croppedNegatives)
    img = double(croppedNegatives{:,i});
    croppedNegImages(:,:,i) = img;
end

npos=size(croppedPosImages,3);
nneg=size(croppedNegImages,3);
% normalize
normPosImages=meanvarpatchnorm(croppedPosImages);
normNegImages=meanvarpatchnorm(croppedNegImages);

% make a vector of image labels
Ypos=ones(npos,1);
Yneg=-ones(nneg,1);

% flatten training images into one vector per sample
xsz=size(normPosImages,2);
ysz=size(normPosImages,1);
Xpos=transpose(reshape(normPosImages,ysz*xsz,npos));
Xneg=transpose(reshape(normNegImages,ysz*xsz,nneg));
% 
croppedImages = [Xpos; Xneg];

% Call=[1000 100 10 1 .1 .01 .001 .0001 .00001];
% indices = (1:size(croppedImages))';
% croppedY = [Ypos; Yneg];
% orderedArray = horzcat(indices,croppedImages, croppedY);
% shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);
% croppedImages = shuffledArray(:,1:end-1)'; 
% croppedY = shuffledArray(:,end);
% [ev, param] = cross_validate(croppedImages,croppedY,ks,n,classifier);
% if (strcmp(classifier,'SVM'))
%     [maxev, maxind] = max(ev);
%     cross_validation_parameter = Call(maxind(1));
%     fprintf('Optimal C=%d\n', cross_validation_parameter);
%     WRbest =param(maxind(1),:).W;
%     BRbest = param(maxind(1),:).B;
% end
% save('cropped_images_train.mat', 'WRbest', 'BRbest');
Call=[.001 .0001 .00001];
indices = (1:size(croppedFeatures))';
croppedY = [Ypos; Yneg];
orderedArray = horzcat(indices,croppedFeatures, croppedY);
shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);
croppedFeatures = shuffledArray(:,1:end-1)'; 
croppedY = shuffledArray(:,end);
[ev, param] = cross_validate(croppedFeatures,croppedY,ks,n,classifier);
if (strcmp(classifier,'SVM'))
    [maxev, maxind] = max(ev);
    cross_validation_parameter = Call(maxind(1));
    fprintf('Optimal C=%d\n', cross_validation_parameter);
    Wbest =param(maxind(1),:).W;
    Bbest = param(maxind(1),:).B;
end
save('cropped_features_train.mat', 'Wbest', 'Bbest');
end

