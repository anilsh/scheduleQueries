
tracker_defaults;
datasets_defaults;

showtext('Running MCT experiments ...');

if exist('configuration') ~= 2
	showtext('Please configure configuration.m.');
	error('Setup file does not exist.');
end;

configuration;

mkpath(track_properties.directory);

%datasets_directory = fullfile(track_properties.directory, 'NLPR_MCT_Dataset');
datasets_directory = fullfile(track_properties.directory, 'MCT dataset');
results_directory = fullfile(track_properties.directory, 'Results');

datasets = load_datasets(datasets_directory);

if size(datasets) ~= datasets_properties.num
	error('No datasets available. Stopping.');
end;

showtext('Preparing %s ...', team_name);

tracker = create_tracker(team_name, tracker_path, ...
    fullfile(results_directory, team_name), 'linkpath', tracker_linkpath);

