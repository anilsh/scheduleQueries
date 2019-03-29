function print_detail(text, varargin)

global track_properties;

if ~track_properties.debug
    return;
end;

showtext(text, varargin{:});
