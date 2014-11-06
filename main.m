%Get histogram data for all images in images folder
imdb = constructIMDB2('data/images', 145, 'image', 2048);
%format the retrieved information for use by object categorization
histograms=zeros(2048, 145);
names = transpose(imdb.images.name);
%go through all images to create the matching histogram
for i=1:45
    row = imdb.images.words{i};
    [~, rowSize] = size(row);
    for j=1:rowSize
        histograms(row(j), i) = histograms(row(j), i) + 1;
    end
end
save('data/testdata.mat', 'histograms', 'names');

%Get histogram data for all images in training images folder
imdb = constructIMDB2('data/myImages', 9, 'bear', 2048);
%format the retrieved information for use by object categorization
histograms=zeros(2048, 9);
names = transpose(imdb.images.name);
%go through all images to create the matching histogram
for i=1:9
    row = imdb.images.words{i};
    [~, rowSize] = size(row);
    for j=1:rowSize
        histograms(row(j), i) = histograms(row(j), i) + 1;
    end
end
save('data/testdata2.mat', 'histograms', 'names');

%Get a few negative images (manually added and removed)
imdb = constructIMDB2('data/myImages', 13, 'image', 2048);
%format the retrieved information for use by object categorization
histograms=zeros(2048, 13);
names = transpose(imdb.images.name);
%go through all images to create the matching histogram
for i=1:13
    row = imdb.images.words{i};
    [~, rowSize] = size(row);
    for j=1:rowSize
        histograms(row(j), i) = histograms(row(j), i) + 1;
    end
end
save('data/testdata3.mat', 'histograms', 'names');