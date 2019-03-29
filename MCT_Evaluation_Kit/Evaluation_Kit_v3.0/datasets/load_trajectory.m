function [trajectory] = load_trajectory(filename)

trajectory = [];

if exist(filename, 'file')
    trajectory_fp = fopen(filename);
    trajectory_tmp = fscanf(trajectory_fp,'%d,%d,%d,%d,%d,%d,%d',[7,inf]);
    trajectory = trajectory_tmp';
    fclose(trajectory_fp);

    [n_frames, n_values] = size(trajectory);

    if n_values ~= 7
        trajectory = [];
        print_detail('WARNING: File "%s" not valid.', filename);
    end;

else
    print_detail('WARNING: File "%s" does not exists.', filename);
end;
