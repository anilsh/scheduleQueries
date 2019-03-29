function [report_file] = get_report(tracker, finished_scores, scores)

global datasets_properties;

report_file = fullfile(tracker.directory, sprintf('%s-%s.html', tracker.identifier, datestr(now, 30)));
columnLabels_names = {'g','r','m','fp','mme_s','mme_c', ...
    'tp_s','tp_c','precision','recall','MCTA'};
temporary_dir = tempdir;
temporary_file = fullfile(temporary_dir, 'tables.tmp');
fid = fopen(temporary_file, 'w');

for i = 1:3
    if finished_scores(i) == 1
    	fprintf(fid, '<h2>Experiment <em>%d</em></h2>\n', i);
        score = [];
        for j = 1:datasets_properties.num
            score_tmp = [scores{i,j}.gt, scores{i,j}.results, scores{i,j}.missing, scores{i,j}.falsepos, ...
                scores{i,j}.mismatch_s, scores{i,j}.mismatch_c, scores{i,j}.truepos_s, scores{i,j}.truepos_c, ...
                scores{i,j}.precision, scores{i,j}.recall, scores{i,j}.mcta];
            score = [score;score_tmp];
        end;
        gethtml(score, fid, 'rowLabels', datasets_properties.name, 'columnLabels', columnLabels_names);
    end;
end;

fclose(fid);

template = fileread(fullfile(fileparts(mfilename('fullpath')), 'results_template.html'));

report = strrep(template, '{{body}}', fileread(temporary_file));

report = strrep(report, '{{tracker}}', tracker.identifier);

report = strrep(report, '{{timestamp}}', datestr(now, 31));

fid = fopen(report_file, 'w');

fwrite(fid, report);

fclose(fid);

showtext('The result file is created!');
%delete(temporary_file);
