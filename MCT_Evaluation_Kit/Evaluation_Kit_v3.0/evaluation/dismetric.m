%%% This function compute the score between groundtruth and detection
function [score ] = dismetric( groundtruth, trackresult, type)
if strcmp(type,'OVERLAP')
    for i = 1:size(groundtruth,1)
        for j = 1:size(trackresult,1)
            score(i,j) = overlapmetric( groundtruth(i,:), trackresult(j,:));
        end
    end
end

end

%% compute a label and a detection
function [score] = overlapmetric( idxLabelData, idxTrackData)
    x1 = idxLabelData(1,4);
    y1 = idxLabelData(1,5);
    w1 = idxLabelData(1,6);
    h1 = idxLabelData(1,7);
    x2 = idxTrackData(1,4);
    y2 = idxTrackData(1,5);
    w2 = idxTrackData(1,6);
    h2 = idxTrackData(1,7);
    
    interarea = rectint([ x1 y1 w1 h1 ],[ x2 y2 w2 h2 ]);
    
    unionarea = w1*h1 + w2*h2 - interarea;
    
    if unionarea == 0
        score = 0;
    else
        score = interarea/unionarea;
    end
    
end