function [isPresent, bbox, gallery] = isTargetPresent_aamas(ped,c,fno, embedGT,gallery, opts)
%
% embedGT is the list of embedding of bboxes at all times as in GT
% gallery is the last seen features of the current target
%

% Check if the selected camera is Cx
if c == opts.num_camera+1
    isPresent = 0;
    bbox = [];
    return;
end


%% Compute distance

% get index of bboxes available at currently selected frame
index_of_bbox = ped(:,1)==c &  ped(:,2)==fno;
% get the feature embedding and corresponding bounding boxes
feat_selected = embedGT(index_of_bbox,:);
bboxes = ped(index_of_bbox,3:6);

% check if bboxes are available at current frame
[num_bbox,~] = size(bboxes);
if (num_bbox<=0)
    isPresent = 0;
    bbox=[]; 
    return;
else
    feats = feat_selected;
end

%% Only distance from existing embeddings of the target
[~,bboxI,gallery] = only_distance_based(gallery, feats);
if bboxI == -1
    isPresent = 0;
    bbox = [];
else
    isPresent = 1;
    bbox = bboxes(bboxI,:);
end

%% compute distance between the gallery embeddings and current frame embeddings
% %score = sqdist(Hist_test, ff);
% score = sqdist(gallery, feats);
% bbox = [];
% predID = [];
% 
% % for each bbox, find the best matching person 
% for i= 1:size(score,2)
%     dist = score(:, i);
%     [~, index] = sort(dist, 'ascend');
%     predID = [predID; testID(index(1))];  % best matching ID is kept
% %     predID = [predID; testID(index(1)),testID(index(2)),testID(index(3)),...
% %         testID(index(4)),testID(index(5)),testID(index(6)),testID(index(7)),...
% %         testID(index(8)),testID(index(9)),testID(index(10)),testID(index(11)),...
% %         testID(index(12)),testID(index(13)),testID(index(14)),testID(index(15))];  % store top rank
% end
% 
% % find if pid is present
% ifPIDPresent = 10000*ones(size(predID,1),1);
% for i = 1:size(predID)
%     if any(pid==predID(i,:))
%         a = find(pid==predID(i,:));
%         ifPIDPresent(i) = a(1);
%     end
% end
% 
% % find min index for pid
% if all(ifPIDPresent == 10000)
%     % return defaults
% else
%     [~,idx] = min(ifPIDPresent );
%     isPresent = 1;
%     bbox = b_box(idx(1),:);
% end

function [p,bboxI,gallery] = only_distance_based(gallery, feats)
thres = 0.3;

score = pdist2(gallery, feats);
dist = mean(score,1);
% select minimum
[minV,minI] = min(dist);

if minV < thres
    p = 1;
    bboxI = minI;
else
    p = 0;
    bboxI = -1;
end

% update gallery set
if minV < 0.2
   gallery = [gallery; feats(bboxI,:)]; 
end
