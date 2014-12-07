function [ img_name, present, location ] = bearTest( image_folder, classifier, feature )
%default classifier and features used for training that will also be used
%for testing our images here
if nargin < 3
    feature = 'DeCAF';
end
if nargin < 2
    classifier = 'SVM';
end
if nargin < 1
    image_folder= 'TestImages';
end
%read in current image
filepath = strcat(image_folder, '/*.jpg');
TestImages = dir(filepath);
img_name = {TestImages.name};


num_images = size(TestImages,1);

%decaf features stored in test.mat same size and order as test images
%so each row in test.mat is the decaf features for each test image
load('trained.mat');
if strcmp(feature, 'DeCAF')
    load('test.mat');
    len_feat = size(decaf_fv,2);
    features = zeros(num_images, len_feat);
    for i=1:num_images
        features(i,:) = decaf_fv(i,:);
    end
end

if strcmp(classifier, 'SVM')
    BearScore = zeros(num_images,1);
    for i=1:num_images
        BearScore(i)= dot(Wbest,features(i,:)) + Bbest;
    end
    save('bearnecessities.mat','BearScore');
end

%ADD STUFF FOR IF WE DON'T DO SVM
%else
 %   K= k;
load('cropped_images_train.mat', 'WRbest', 'BRbest');
wHOG = load('cropped_features_train.mat');
wHOG = wHOG.Wbest;
bHOG = load('cropped_features_train.mat');
bHOG = bHOG.Bbest;
load('cropped_lbp_train.mat');
wLBP = load('cropped_features_train.mat');
wLBP = wLBP.Wlbp;
bLBP = load('cropped_features_train.mat');
bLBP = bLBP.Blbp;
present = zeros(num_images,1);
for i=1:num_images
   if BearScore(i) < 0
       fprintf('no bear\n');
       
   end
   if BearScore(i) >= 0
       fprintf('bear found\n');
       present(i) = 1;
       
       % %do localization if bear in image
       img = imread([image_folder '/' TestImages(i).name]);
       threshold = [-1.1, -1.19,-1];
       scale = [6/17, 1/4,1/4];
       
       numScales = size(scale,2);
       
       for z=1:numScales
           I = imresize(img, scale(z));


           % compute Wbest on raw pixels
           S = imfilter(double(I),WRbest,'same','corr','replicate');
           % do non-maximum suppression
           Sdilated = imdilate(S,strel('disk',24));
           B = Sdilated==S;
           % find out face location centers
           [y,x] = find(B);
           maxX = size(I, 2);
           maxY = size(I, 1);
           %bbox=[x(:)-24/2 y(:)-24/2 x(:)+24/2 y(:)+24/2];
           r = 1;
           scores = [];
           y_box = [];
           x_box = [];
           for j=1:size(x,1)
               if(x(j)>12 && y(j)>12 && x(j)<maxX-12 && y(j)<maxY-12)
                   yTL = y(j)-12;
                   yBR = y(j)+12;
                   xTL = x(j)-12;
                   xBR = x(j)+12;
                   segment= single(I(yTL:yBR, xTL:xBR));
                   HoG=vl_hog(segment,24);
                   HoG = squeeze(HoG)';
                   HoG = HoG(:,2:end);
                   prediction = dot(wHOG, HoG) + bHOG;

                   if prediction > threshold(z)
                       LBP = vl_lbp(segment,24);
                       LBP = squeeze(LBP)';
                       prediction = dot(wLBP(:,2:end), LBP) + bLBP;
                       scores(r) = prediction;
                       x_box(r) = x(j);
                       y_box(r) = y(j);
                       r = r + 1;
                   end
               end
           end

           numboxes=size(x_box,2);
           [maxScore, maxInd] = max(scores);
           x = x_box(maxInd);
           y = y_box(maxInd);
           location(i,:) = [x,y,24,24];
           figure, imshow(I);
           hold on
           rect = rectangle('position',location(i,:),'EdgeColor', [0 1 0]);

           hold off

       end
   end
       
end
    
 
 

% 

    
end

% This is a template file for your test program for the course project
%%%%%%%%%%%%%%% Input arguments %%%%%%%%%%%%%%%%%%%%%%%
% input_image_file: is the file name of the input image, so your program
% will be able to read the image with I = imread(input_image_file);
%%%%%%%%% Output arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% img_name : name of the image (with extension) whose result is provided
% present : 0 if bear not present, 1 otherwise
% location : only needed if present=1: vector of length 4 that contains
%           the x,y coordinates of the top left corner of the bounding box
%            followed by its width and height organized as:
%               [top_left_x, top_left_y, width, height]
%           This assumes that the origin is at the top left corner of the
%           image.


