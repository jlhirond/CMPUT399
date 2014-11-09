%Main takes no input parameters, main runs constructIMDB2 three times to
%construct three sets of data for use by compareImages.m
%The first data set is for all test images
no_of_words = 2048;
%Value from 0 - 1 that resizes the image. Ex. 0.5 reduces the size by 2x.
imageSize=0.5;

%Get histogram data for all negative images in images folder
image_count = 225;
imdb = constructIMDB2('data/images', image_count, 'image', no_of_words, imageSize);
%format the retrieved information for use by object categorization
histograms=zeros(no_of_words, image_count);
names = transpose(imdb.images.name);
%go through all images to create the matching histogram
for i=1:image_count
    row = imdb.images.words{i};
    [~, rowSize] = size(row);
    for j=1:rowSize
        histograms(row(j), i) = histograms(row(j), i) + 1;
    end
end
save('data/negdata.mat', 'histograms', 'names');

%Get histogram data for all positive images in images folder
image_count = 10;
imdb = constructIMDB2('data/images', image_count, 'image (30a)', no_of_words, imageSize);
%format the retrieved information for use by object categorization
histograms=zeros(no_of_words, image_count);
names = transpose(imdb.images.name);
%go through all images to create the matching histogram
for i=1:image_count
    row = imdb.images.words{i};
    [~, rowSize] = size(row);
    for j=1:rowSize
        histograms(row(j), i) = histograms(row(j), i) + 1;
    end
end
save('data/posdata.mat', 'histograms', 'names');
