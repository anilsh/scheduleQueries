function [experiment_index] = select_experiment()

showtext('Please choose an Experiment:');
showtext('1 - Experiment One.');
showtext('2 - Experiment Two.');
showtext('3 - Experiment Three.');

option = input('Selected an Experiment: ', 's');

experiment_index = int32(str2double(option));

if isempty(experiment_index) || experiment_index < 1 || experiment_index > 3
    experiment_index = [];
    showtext('The selected Experiment is invalid.');
end;