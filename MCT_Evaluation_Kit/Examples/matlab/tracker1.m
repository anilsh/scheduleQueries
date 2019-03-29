function [status] = tracker1()

status = 1;

input_groundtruth = [];
dataset = [];

if exist('input_groundtruth.txt','file')
	input_groundtruth = 0;%csvread('input_groundtruth.txt');
end;

if exist('dataset.txt','file')
	dataset = 0;%csvread('dataset.txt');
end;

%read result.dat just for testing, please replace by your algorithm
output = example_tracker(input_groundtruth,dataset);
%algorithm done

if exist('output', 'var')
    csvwrite('output.txt',output);
    status = 0;
end;