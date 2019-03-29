
global track_properties;

% track_properties.directory = '...\workingdirectory';
%track_properties.directory = 'D:\workingdirectory';
track_properties.directory = '/media/win/data';

% Set 1 for tracker.exe. Set to 0 for tracker.m. 
track_properties.execute = 0;

% Set 1 when running the example tracker. And set to 0 when running your trackers. 
track_properties.example = 0;

% Set 1 if needing more details of debug.
track_properties.debug = 1;

% team_name = 'Example_Tracker';
team_name = 'MAP-test';

tracker_path = '/home/anils/Downloads/MCT_Evaluation_Kit/Examples/matlab';
%tracker_path = 'D:\workingdirectory\tracker\MAP-test';
%tracker_path = '/home/anils/Notebook/camera/MultiCam';


tracker_linkpath = {'/home/anils/Downloads/MCT_Evaluation_Kit/Examples/matlab'};
%tracker_linkpath = {'D:\workingdirectory\tracker\MAP-test'};
%tracker_linkpath = {'/home/anils/Notebook/camera/MultiCam'};

