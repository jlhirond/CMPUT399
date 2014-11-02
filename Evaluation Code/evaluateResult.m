function score=evaluateResult(img_name, present, location)
% img_name : name of the image (with extension) whose result is provided
% present : 0 if bear not present, 1 otherwise
% location : only needed if present=1: vector of length 4 that contains
%           the x,y coordinates of the top left corner of the bounding box
%            followed by its width and height organized as:
%               [top_left_x, top_left_y, width, height]
%           This assumes that the origin is at the top left corner of the
%           image.
% score: real number between 0 and 1 indicating the degree to which the
%       result bounding box matches the ground truth; will be 0 if
%       result.present is incorrect
% If the bear is actually present in the image, it also displays the locations of the 
% two bounding boxes as semi-transparent rectangles, correct one in green 
% and provided one in red with overlapping portions in yellow (assuming 
% that present=1 too).
% Before calling this function, please ensure that all negative images are in a
% single folder called 'NegativeImages'

if nargin<3
    present=0;
    location=zeros(1,4);
end
result.present=present;
result.location=location;

pos_dir='PositiveImages';
neg_dir='NegativeImages';

pos_struct=dir(pos_dir);
neg_struct=dir(neg_dir);

pos_count=length(pos_struct)-2;
neg_count=length(neg_struct)-2;

pos_images=cell(pos_count, 1);
for i=1:pos_count
    pos_images{i}=pos_struct(i+2).name;
end

neg_images=cell(neg_count, 1);
for i=1:neg_count
    neg_images{i}=neg_struct(i+2).name;
end
pos_index=find(ismember(pos_images, img_name));
if ~isempty(pos_index)
    ground_truth.present=1;
    ground_truth.location=getLocation(pos_index-1, pos_images);
    img=imread(sprintf('%s/%s', pos_dir, img_name));
    
    binary_mask_truth=zeros(size(img));
    top_x=ground_truth.location(1);
    top_y=ground_truth.location(2);
    bottom_x=top_x+ground_truth.location(3);
    bottom_y=top_y+ground_truth.location(4);
    binary_mask_truth(top_y:bottom_y, top_x:bottom_x)=1;
    
    img_copy=img(:, :, :);
    img_copy(top_y:bottom_y, top_x:bottom_x, 2)=img_copy(top_y:bottom_y, top_x:bottom_x, 2)+40;
    
elseif ~isempty(find(ismember(neg_images, img_name), 1))
    ground_truth.present=0;
else
    error('Invalid image name provided');
end

if result.present~=ground_truth.present
    score=0;
else
    if ground_truth.present==0
        score=1;
        return;
    end
    binary_mask_res=zeros(size(img));
    top_x=result.location(1);
    top_y=result.location(2);
    bottom_x=top_x+result.location(3);
    bottom_y=top_y+result.location(4);
    binary_mask_res(top_y:bottom_y, top_x:bottom_x)=1;
    img_copy(top_y:bottom_y, top_x:bottom_x, 1)=img_copy(top_y:bottom_y, top_x:bottom_x, 1)+40;
    figure, imshow(img_copy);
    union_count=numel(find(binary_mask_truth | binary_mask_res));
    inter_count=numel(find(binary_mask_truth & binary_mask_res));
    score=inter_count/union_count;
end

end

