function [working_directory] = prepare_input(dataset, experiment_index)

working_directory = tempname;
mkdir(working_directory);

input_groundtruth_file = fullfile(working_directory, 'input_groundtruth.txt');
dataset_file = fullfile(working_directory, 'dataset.txt');
input_groundtruth=[];

switch experiment_index
    case 1 
        cams=size(dataset.directory,1);
        for cam=1:cams
            input_groundtruth_tmp = dataset.groundtruth( find(dataset.groundtruth(:,1) == cam),:);
            new_id = input_groundtruth_tmp(:,3);
            [new_ord, old_position] = sort(new_id,1);
            new_id(old_position(1)) = 0;
            %new_id(1) = 0;
            ord = 0;
            for i=2:size(new_id)
                if new_ord(i)~=new_ord(i-1)
                    ord = ord+1;
                end
                new_id(old_position(i)) = ord;
                %new_id(i) = ord;
            end
            input_groundtruth_tmp(:,3) = new_id;
            input_groundtruth = [input_groundtruth;input_groundtruth_tmp];
        end;        
        csvwrite(input_groundtruth_file, input_groundtruth);
    case 2 
        input_groundtruth = dataset.groundtruth;
        input_groundtruth(:,3) = 0;
        csvwrite(input_groundtruth_file, input_groundtruth);
    case 3 
        input_groundtruth = dataset.groundtruth;
        input_groundtruth(:,1:7) = 0;
        csvwrite(input_groundtruth_file, input_groundtruth);
end 

dataset_fp = fopen(dataset_file, 'w');
for i = 1:size(dataset.directory)
    fprintf(dataset_fp, '%s\n', dataset.directory{i});
end;
fclose(dataset_fp);

