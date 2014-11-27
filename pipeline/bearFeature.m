 function feature = bearFeature(image, type)
 if strcmp(type, 'hist')
     image = rgb2gray(image);
     h=imhist(image, 32);
     feature=h(:)./sum(h);
 end
 end