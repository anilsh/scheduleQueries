function showtext(text, varargin)

global track_properties;

if ispc
    text = strrep(text, '\', '\\');
end

if nargin > 1
    fprintf([text, '\n'], varargin{:});
else
    fprintf([text, '\n']);
end;

if is_octave()
    fflush(stdout);
else
    drawnow('update');
end;
