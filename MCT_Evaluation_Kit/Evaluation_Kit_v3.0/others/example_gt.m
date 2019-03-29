function [gt] = example_gt(directory)

if exist(directory, 'file')
	example_gt_fp = fopen(directory);
	example_gt_tmp = fscanf(example_gt_fp,'%d %d %d %d %d %d %d',[7,inf]);
	gt = example_gt_tmp';
	fclose(example_gt_fp);
else
    showtext('Can not find gt.dat file.');
    gt = [];
end;
