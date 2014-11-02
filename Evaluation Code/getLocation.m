function [location] = getLocation(img_index, img_list)
% extracts corner coordinates from the filename in  img_list with the given
%  index and having the prefix 'boxed'

img_prefix=sprintf('boxed_%04d', img_index);
img_id=~cellfun(@isempty,strfind(img_list, img_prefix));
img_name=img_list{img_id};
div_indexes=strfind(img_name,'_');
if length(div_indexes)~=5
    error('Unexpected filename for extracting location: %s\n',img_name);
end
end_index=strfind(img_name,'.jpg');
div_indexes=[div_indexes(2:end) end_index];
location=zeros(1, 4);
for i=1:4
    coord_str=img_name(div_indexes(i)+1:div_indexes(i+1)-1);
    location(i)=str2num(coord_str);
end



