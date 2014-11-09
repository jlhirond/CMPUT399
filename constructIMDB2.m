function [imdb] = constructIMDB(dir_name, count, template, no_of_words, imageSize)
% reads images from the given directory and constructs a database
run('VLFEATROOT/toolbox/vl_setup')

% default parameters
% if nargin<4
%     no_of_words=2048;
% end
% if nargin<3
%     template='bear';
% end
% if nargin<2
%     count=45;
% end
% if nargin<1
%     dir_name='data/images';
% end
start_stage=1; % set this to an integer <=5 to run only part of the function 
% while loading the results of earlier stages from a mat file saved 
% by a previous run
build_vocab=1;% set this to 0 to load the vocabulary from external source
% following is the file from which to load the vocaabulary and KD tree if 
% build_vocab=0
save_template='loop_closure_imdb';

if start_stage<=1
    % Stage 0: Initialize imdb structure
    images.name=cell(1, count);
    images.id=1:count;
    images.frames=cell(1, count);
    images.words=cell(1, count);
    images.descrs=cell(1, count);
    imdb.images=images;
    imdb.dir=dir_name;
    imdb.featureOpts={'method','dog','affineAdaptation',true,'orientation',false};
    imdb.numWords=no_of_words;
    imdb.sqrtHistograms=0;
    
    % Stage 1: Read images and extract features
    disp('Reading images and extracting features....');
    tic
    for i=1:count
        img_name=sprintf('%s (%d).jpg', template, i);
        img=imread(strcat(dir_name, '/', img_name));
        img=imresize(img, imageSize);
        [frames, descrs] = getFeatures(img, 'peakThreshold', 0.001, 'orientation', false);
        imdb.images.name{i}=img_name;
        imdb.images.frames{i}=frames;
        imdb.images.descrs{i}=descrs;
        if mod(i, 10)==0
            fprintf('Processed %d of %d images\n', i, count);
        end
    end
    toc
    %save(strcat(save_template, '1.mat'), 'imdb');
elseif start_stage==2
    load(strcat(save_template, '1.mat'));
end

if start_stage<=2
    % Stage 2: Construct a vocabulary of words from the combined features of all the images
    % using K-Means clustering (or load it from external source)
    if build_vocab
        fprintf('Constructing a vocabulary of %d words....\n', no_of_words);
        tic
        % descrs_combined=cellArrayToMatrix(imdb.images.descrs);
        descrs_combined=cat(2,imdb.images.descrs{:}); % concatenate the features
        % from all images into a single array
        [imdb.vocab, imdb.assignments]=vl_kmeans(descrs_combined, no_of_words, 'algorithm', 'ELKAN');
        toc
    end        
    %save(strcat(save_template, '2.mat'), 'imdb');
elseif start_stage==3
    load(strcat(save_template, '2.mat'));
end

if start_stage<=3
    % Stage 3: Construct KD tree from this vocabulary (or load it from 
    % external source)
    if build_vocab        
        disp('Constructing KD tree from the vocabulary....');
        tic
        imdb.kdtree=vl_kdtreebuild(imdb.vocab);
        toc
    end
    %save(strcat(save_template, '3.mat'), 'imdb');
elseif start_stage==4
    load(strcat(save_template, '3.mat'));
end
if isa(imdb.vocab, 'double')
    double_vocab=1;
elseif isa(imdb.vocab, 'single')
    double_vocab=0;
else
    error('Vocabulary has invalid data type: %s',class(imdb.vocab));
end
if start_stage<=4
    % Stge 4: Find the words present in each image through NN search of the
    % vocabulary
    disp('Getting words from features....');
    tic
    for i=1:count
        current_descrs=imdb.images.descrs{i};
        if double_vocab
            current_descrs=double(current_descrs);
        else
            current_descrs=single(current_descrs);
        end
        [imdb.images.words{i}, ~]=vl_kdtreequery(imdb.kdtree,imdb.vocab,...
            current_descrs, 'MaxNumComparisons', 1000);
        if mod(i, 10)==0
            fprintf('Processed %d of %d images\n', i, count);
        end
    end
    toc
    %save(strcat(save_template, '.mat'), 'imdb');
else
    load(strcat(save_template, '.mat'))
end

end

