function [datasets] = load_datasets(directory)

%global track_properties;
global datasets_properties;

datasets = cell(0);

mkpath(directory);

for i=1:datasets_properties.num
    %video_directory = fullfile(directory, 'video', char(datasets_properties.name(i)));
    video_directory = fullfile(directory, char(datasets_properties.name(i)));
    %%groundtruth_directory = fullfile(directory, 'annotation', char(datasets_properties.name(i)));
    groundtruth_directory = fullfile(directory, 'annotation_files/annotation', char(datasets_properties.name(i)));

    if ~exist(video_directory, 'dir')|| ~exist(groundtruth_directory, 'dir')
        continue;
    end;

    datasets{end+1} = create_dataset(i, video_directory, groundtruth_directory);

end;
