function [output] = example_tracker(input_groundtruth,dataset)

output = [];

if exist('result.dat','file')
    output_tp = fopen('result.dat');
    output_tmp = fscanf(output_tp,'%d %d %d %d %d %d %d',[7,inf]);
    output = [output;output_tmp'];
    fclose(output_tp);
end;