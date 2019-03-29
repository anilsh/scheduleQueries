function [selected_dataset] = select_dataset(datasets)

global datasets_properties;

showtext('Please choose a dataset:');

for i = 1:datasets_properties.num
    showtext('%d - Dataset%d with %d cameras.', i, i, datasets_properties.cams(i));
end;

option = input('Selected dataset: ', 's');

selected_dataset = int32(str2double(option));

if isempty(selected_dataset) || selected_dataset < 1 || selected_dataset > datasets_properties.num
    selected_dataset = [];
    showtext('The selected dataset is invalid.');
end;