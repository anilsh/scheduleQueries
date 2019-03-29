function gethtml(matrix, filename, varargin)

    rowLabels = [];
    colLabels = [];
    format = [];
    css_class = [];
    if (rem(nargin,2) == 1 || nargin < 2)
        error('Incorrect number of arguments to %s.', mfilename);
    end

    okargs = {'rowlabels','columnlabels', 'format', 'class'};
    for j=1:2:(nargin-2)
        pname = varargin{j};
        pval = varargin{j+1};
        k = strmatch(lower(pname), okargs);
        if isempty(k)
            error('Unknown parameter name: %s.', pname);
        elseif length(k)>1
            error('Ambiguous parameter name: %s.', pname);
        else
            switch(k)
                case 1  % rowlabels
                    rowLabels = pval;
                    if isnumeric(rowLabels)
                        rowLabels = cellstr(num2str(rowLabels(:)));
                    end
                case 2  % column labels
                    colLabels = pval;
                    if isnumeric(colLabels)
                        colLabels = cellstr(num2str(colLabels(:)));
                    end
                case 3  % format
                    format = lower(pval);
                case 4  % format
                    css_class = pval;
            end
        end
    end

    if (ischar(filename))
        fid = fopen(filename, 'w');
        close_file = 1;
    else
        fid = filename;
        close_file = 0;
    end;
    
    width = size(matrix, 2);
    height = size(matrix, 1);

    if isnumeric(matrix)
        matrix = num2cell(matrix);
        for h=1:height
            for w=1:width
                if(~isempty(format))
                    matrix{h, w} = num2str(matrix{h, w}, format);
                else
                    matrix{h, w} = num2str(matrix{h, w});
                end
            end
        end
    end
    
    if(~isempty(css_class))
        fprintf(fid, '<table class="%s">\n', css_class);
    else
        fprintf(fid, '<table border="2">\n');
    end
    
    if(~isempty(colLabels))
        fprintf(fid, '<tr>');
        if(~isempty(rowLabels))
            fprintf(fid, '<th><center>&nbsp;</center></th>');
        end
        for w=1:width
            fprintf(fid, '<th width="100px"><center>%s</th>', colLabels{w});
        end
        fprintf(fid, '</tr>\r\n');
    end
    
    for h=1:height
        fprintf(fid, '<tr>');
        if(~isempty(rowLabels))
            fprintf(fid, '<th width="200px"><center>%s</center></th>', rowLabels{h});
        end
        for w=1:width
            fprintf(fid, '<th><center>%s</center></th>', matrix{h, w});
        end
        fprintf(fid, '</tr>\r\n');
    end

    fprintf(fid, '</table>\r\n');
    
    if (close_file)
        fclose(fid);
    end;
