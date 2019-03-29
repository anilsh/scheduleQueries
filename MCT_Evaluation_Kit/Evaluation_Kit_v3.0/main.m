clc;
clear;

script_directory = fileparts(mfilename('fullpath'));
include_dirs = cellfun(@(x) fullfile(script_directory,x), {'', 'datasets', ...
    'evaluation', 'others', 'tracker'}, 'UniformOutput', false); 
addpath(include_dirs{:});

initialize;

global current_dataset;
global datasets_properties;
global track_properties;

if ~exist('trajectory', 'var')
	trajectory = [];
end;

finished_scores = [0,0,0];
if ~exist('scores', 'var')
	scores = cell(3,datasets_properties.num);
end;

if ~exist('current_dataset', 'var')
	current_dataset = 1;
end;

performance = struct('MCTA', 0);

showtext('');
showtext('***************************************************************************');
showtext('');
showtext('Welcome to the MCT Evaluation Kit!');
showtext('This process will help you evaluate your tracker.');
showtext('a, b, c, d to verify the execution and the output data.');
showtext('');
showtext('***************************************************************************');
showtext('');

while 1
    showtext('Choose an action:');

    showtext('a - Run the tracker on a selected sub-dataset for a selected experiment (optional)');
    showtext('b - Run the tracker on all the sub-datasets for a selected experiment (required)');
    if ~isempty(trajectory)
        showtext('c - Display the latest result compared with the groundtruth (optional)');
    end;
    if sum(finished_scores) ~= 0
        showtext('d - Print results of all the experiments (required)');
    end;
    showtext('e - Exit');

    option = input('Choose an action: ', 's');

    switch option
    case 'a'
        experiment_index = select_experiment();
        current_dataset_index = select_dataset(datasets); 
        
        if ~isempty(current_dataset_index)&&~isempty(experiment_index)
            showtext('Dataset%d is processing...',current_dataset_index);
            %trajectory = tracker.run(tracker, datasets{current_dataset_index}, experiment_index);
                
            if track_properties.example == 1
                example_gt_directory = fullfile(tracker.command,'gt.dat');
                gt = example_gt(example_gt_directory);                    
            else
                gt = datasets{current_dataset_index}.groundtruth;
            end;
            
            %gt = datasets{current_dataset_index}.groundtruth;
            %trajectory = csvread('/media/win/anils/CogVis 2018/MultiCam/results_aamas/MTMC_traj_db4_exp2_gt_aamas');
            res_dir_path = '/media/win/anils/CogVis 2018/MultiCam/results_aamas';
            trajectory = csvread([res_dir_path,'/MTMC_traj_db4_exp1_err20Seq_traj_aamas']);
            %trajectory = csvread('/media/win/anils/CogVis 2018/MultiCam/traj_db4_exp2_allP');
            [gt,trajectory] = arrange_cam_policy(gt,trajectory, current_dataset_index);
            overlap_rate = 0.5;
            evalResult = evalMCTA(trajectory, gt, overlap_rate, experiment_index)
        end;
    case 'b'
        experiment_index = select_experiment();
        if ~isempty(experiment_index)
           for i = 1:datasets_properties.num           
                showtext('Dataset%d is processing...',i);
                trajectory = tracker.run(tracker, datasets{i}, experiment_index);
                
                if track_properties.example == 1
                    example_gt_directory = fullfile(tracker.command,'gt.dat');
                    gt = example_gt(example_gt_directory);                    
                else
                    gt = datasets{i}.groundtruth;
                end;
                        
                overlap_rate = 0.5;
                evalResult = evalMCTA(trajectory, gt, overlap_rate, experiment_index);
                scores{experiment_index,i} = evalResult;
                finished_scores(experiment_index) = 1;
            end;  
        end;
    case 'c'
        if ~isempty(evalResult)
            showEvalResult( evalResult );
        end;
    case 'd'
        if sum(finished_scores) ~= 0
            get_manifest( tracker );
            get_report( tracker, finished_scores, scores );            
        end;
    case 'e'
        break;
    end;    
end;


