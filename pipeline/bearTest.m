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
num_images= size(TestImages,1);




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
%BearScore= Wbest*

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

for i=1:num_images
   if BearScore(i) < 0
       fprintf('no bear\n');
   end
   if BearScore(i) >= 0
       fprintf('bear found\n');
         
       
       %train SVM to get small Wbest
       
       % %do localization if bear in image
       I = imread([image_folder '/' TestImages(i).name]);
       % compute score function by correlating I with filter W
       S = imfilter(I,Wbest,'same','corr','replicate');
       
       % do non-maximum suppression
       Sdilated = imdilate(S,strel('disk',25));
       
       B = Sdilated==S;
       [y,x] = find(B);
       
       numboxes=size([y,x],1);

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


