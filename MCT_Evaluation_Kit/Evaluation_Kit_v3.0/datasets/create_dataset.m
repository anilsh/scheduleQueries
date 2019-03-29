function [dataset] = create_dataset(dataset_index, video_directory, groundtruth_directory)

global datasets_properties;

groundtruth = [];
directory = cell(0);

for i=1:datasets_properties.cams(dataset_index)
    groundtruth_file = fullfile(groundtruth_directory, ['Cam', num2str(i), '.dat']); 
    groundtruth_file_fp = fopen(groundtruth_file);
    groundtruth_tmp = fscanf(groundtruth_file_fp,'%d %d %d %d %d %d %d',[7,inf]);
    groundtruth = [groundtruth;groundtruth_tmp'];
    directory_tmp = fullfile(video_directory, ['Cam', num2str(i), '.avi']);
    directory = [directory;directory_tmp];
    fclose(groundtruth_file_fp);
end

dataset = struct('name', datasets_properties.name{dataset_index}, 'directory', {directory}, 'groundtruth', groundtruth);

if size(dataset.groundtruth) < 1
    error('%s loads groundtruth failed!', name);
end;


