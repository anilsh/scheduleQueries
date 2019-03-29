%% this function is to find the co-pair of detection and groundtruth
%%% dis is the min score to determin a pair
function [pair] = pairfind(score, dis)
    pair = zeros(size(score,1), size(score,2));
    [row height] = size(score);
 %   index = score < dis;             %%find the index of index which suit for score 
 %   score(index) = 0;
    cnt = min(row,height); 
    for i=1:cnt
        [value idxMax] = max(score(:));
        [row,column] = ind2sub(size(score),idxMax);
        if( score(row, column)>= dis)
            score(row,:) = 0;
            score(:,column) = 0;
            pair(row,column) = 1;
        else
            break;
        end
    end
end