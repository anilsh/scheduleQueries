function [trajectory] = run_tracker(tracker, dataset, experiment_index)

global track_properties;

single_gt_name = ['Experiment', int2str(experiment_index), '_', dataset.name, '.txt'];
single_gt_dir = fullfile(track_properties.directory, 'Results', tracker.identifier);

if ~exist(fullfile(single_gt_dir,single_gt_name), 'file')
    
    working_directory = prepare_input(dataset, experiment_index);

    output_file = fullfile(working_directory, 'output.txt');

    library_path = '';

    output = [];

    % run the tracker
    old_directory = pwd;
    
        if track_properties.execute == 0
            addpath(tracker.command);
            execute_dir = fullfile(tracker.command, ['tracker',int2str(experiment_index), '.m']);
            exefunc = cell(1,3);
            exefunc{1} = @tracker1;
            exefunc{2} = @tracker2;
            exefunc{3} = @tracker3;
        else if track_properties.execute == 1
            execute_dir = fullfile(tracker.command, ['tracker',int2str(experiment_index), '.exe']);
            end;
        end;

    if track_properties.example==1
        example_directory = fullfile(tracker.command);
        cd(example_directory);
        copyfile('result.dat',working_directory);
    end;

    try
        print_detail(['INFO: Executing "', execute_dir, '" in "', working_directory, '".']);
        cd(working_directory);

        if is_octave()
            if track_properties.execute == 0
                status = feval(exefunc{experiment_index});                
            else if track_properties.execute == 1
                [status, output] = system(execute_dir, 1);
                end;
            end;
        else
            % Save library paths
            library_path = getenv('LD_LIBRARY_PATH');

            % Make Matlab use system libraries
            if ~isempty(tracker.linkpath)
                userpath = tracker.linkpath{end};
                if length(tracker.linkpath) > 1
                    userpath = [sprintf(['%s', pathsep], tracker.linkpath{1:end-1}), userpath];
                end;
                setenv('LD_LIBRARY_PATH', [userpath, pathsep, getenv('PATH')]);
            else
                setenv('LD_LIBRARY_PATH', getenv('PATH'));
            end;


            if verLessThan('matlab', '7.14.0')
                if track_properties.execute == 0
                    status = feval(exefunc{experiment_index});                
                else if track_properties.execute == 1
                    [status, output] = system(execute_dir);
                    end;
                end;                
            else
                if track_properties.execute == 0
                    status = feval(exefunc{experiment_index});                
                else if track_properties.execute == 1
                    [status, output] = system(execute_dir, '');
                    end;
                end;                   
            end;
        end;

        if status ~= 0 
            print_detail('WARNING: System command has not exited normally.');
        end;

    catch e
        print_detail('ERROR: Exception thrown "%s".', e.message);
    end;

    if ~isempty(output_file)
        copyfile('output.txt',single_gt_name);
        movefile(single_gt_name,single_gt_dir);
    end;

    cd(old_directory);
    
    % validate and process results
    trajectory = load_trajectory(output_file);
    
else
    trajectory = load_trajectory(fullfile(single_gt_dir,single_gt_name));
    track_properties.cleanup = 0;
end;

n_frames = size(trajectory, 1);

if isempty(trajectory)
	error('No result produced by tracker. Stopping.');
end;

if track_properties.cleanup
    removedir(working_directory);
end;

