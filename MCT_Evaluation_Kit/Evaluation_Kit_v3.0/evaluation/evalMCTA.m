function [ evalPerformance ] = evalMCTA( trackData, groundTruth, overlap, experiment_index)
%% initialize parameters
falsepos = 0;
missing = 0;
hypothesis = 0;
groundtruthes = 0;
mismatch_s = 0;
mismatch_c = 0;
truepos_s = 0;
truepos_c = 0;
precision = 1;
recall = 1;

%% check input data
if isempty(trackData)
    showtext('trackData is empty !');
    return;
end
if isempty(groundTruth)
    showtext('groundTruth is empty !')
    return;
end
if isempty(overlap)
    showtext('overlap is empty !')
    return;
end
if max(trackData(:,1)) ~= max(groundTruth(:,1))
    showtext('groundTruth and hypothesis contain different cameras !')
    return;
end
%% only  compute the trackData whos frame is in the groundtruth 
gtFrame = unique( groundTruth(:,2) ); %get the frame numbers of groundtruth,it must be not empty
[frameHeight frameWith] = size( gtFrame ); 

pre_map = [];   %%% record last frame's [groundtruth trackinglabel] pairs
%% get the i frame's label data and tracking data
test1=0;
test2=0;
for indexframe = 1:frameHeight
    idxLabelData = groundTruth(find(groundTruth(:,2) == gtFrame(indexframe)),:);
    idxTrackData = trackData( find( trackData(:,2) == gtFrame(indexframe) ),: );
    %% Tmp Counter
    falsepos_tmp = 0;
    missing_tmp = 0;
    hypothesis_tmp = 0;
    groundtruthes_tmp = 0;
    mismatch_s_tmp = 0;
    mismatch_c_tmp = 0;
    truepos_s_tmp = 0;
    truepos_c_tmp = 0;
	map = [];       %%% record [groundtruth trackinglabel] pairs
%% get the data for each camera
    for indexcam = 1:max(groundTruth(:,1))
        %%% co-pair is a flag matrix with size of [M N], 
        %%% M represent number of groundtruth in frame i
        %%% N represent number of tracking result in frame i
        co_pair = [];   
        score = [];     %%% score is a [M N] distance matrix 
        idxLabelData_cam = idxLabelData( find(idxLabelData(:,1) == indexcam),:);
        idxTrackData_cam = idxTrackData( find( idxTrackData(:,1) == indexcam ),: );
        %% find the co-pair of groundtruth and the detection
        if ~isempty(idxTrackData_cam)&&~isempty(idxLabelData_cam)
            %%%do the greedy alogrithm to find co-pair of groundtruth and detection        
            score = zeros(size(idxLabelData_cam,1), size(idxTrackData_cam,1));
            co_pair = zeros(size(idxLabelData_cam,1),size(idxTrackData_cam,1));
            %%%compute the distance between groundtruth and detection
            score = dismetric(idxLabelData_cam, idxTrackData_cam, 'OVERLAP');
            %%%compute the co-pair
            co_pair = pairfind( score, overlap);     %just compute the co_pair based on score>=overlap
        else %%% no record in tracking
            score   = [];
            co_pair = [];
        end;   
%%  compute the  number of truepos
        if ~isempty(co_pair) 
        [r_index c_index] = find( co_pair == 1);
            %%% compute the map of gt-tracking in current frame
            for i = 1:length(r_index)
                map = [map; idxLabelData_cam(r_index(i),3), idxTrackData_cam(c_index(i), 3), ...
                    idxLabelData_cam(r_index(i),1)];%indexcam
            end;
        end;
    end;
%% compute the difference in map and pre-map
    hypothesis_tmp = size(idxTrackData,1);
    groundtruthes_tmp = size(idxLabelData,1);
    missing_tmp = groundtruthes_tmp - size(map,1);
    falsepos_tmp = hypothesis_tmp - size(map,1);
    
    %%% compute the difference in map and pre-map
    if ~isempty(map)
        repeats_c=[];
        repeats_s=[];
        for i=1:size(map,1)
            idswitchflag_s = 0;
            idswitchflag_c = 0;
            trueposflag_s = 0;
            old_id=[];
            cur_id=[];
            rep_id_s=[];
            rep_id_c=[];
            if ~isempty(pre_map)
                id = find(pre_map(:,1) == map(i,1));
                if isempty(id)
                    old_id = find(pre_map(:,2) == map(i,2));
                    if ~isempty(old_id)
                        idswitchflag_c = 1;%new target with a old ID.
                    else
                        cur_id = find(map(:,2) == map(i,2));
                        if ~isempty(cur_id)&&size(cur_id,1)~=1
                            idswitchflag_c = 1;%new target with a current repeat ID.
                            idd=[];
                            for k=1:size(cur_id,1)
                                idd = find(pre_map(:,1)==map(cur_id(k),1));
                                idd1 = find(pre_map(idd,2)==map(cur_id(k),2));
                                idd2 = find(pre_map(idd(idd1),3)==map(cur_id(k),3));
                            end;
                            if isempty(idd2)
                                repeats_c=[repeats_c;map(i,2)];
                            end;
                        end;
                    end;
                else
                    if pre_map(id,3) == map(i,3)
                    	trueposflag_s = 1;
                    end;
                    if pre_map(id,2) ~= map(i,2)
                        if pre_map(id,3) == map(i,3)
                            idswitchflag_s = 1;%old target with a wrong ID in single camera
                            %test2=test2+1;
                        else
                            idswitchflag_c = 1;%old target with a wrong ID across cameras
                        end;
                    else                        
                        cur_id = find(map(:,2) == map(i,2));
                        if pre_map(id,3) == map(i,3)
                            rep_id_s = find(map(cur_id,3) == map(i,3));
                            if ~isempty(rep_id_s)&&size(rep_id_s,1)~=1
                                idswitchflag_s = 1;%two target with the same ID in single camera
                                repeats_s=[repeats_s;map(i,2),map(i,3)];
                                %test1=test1+1;
                            end;
                        else
                            if ~isempty(cur_id)&&size(cur_id,1)~=1
                                idswitchflag_c = 1;%two target with the same ID across cameras
                                idd=[];
                                for k=1:size(cur_id,1)
                                    idd1 = find(pre_map(:,1)==map(cur_id(k),1));
                                    idd2 = find(pre_map(idd1,2)==map(cur_id(k),2));
                                    idd3 = find(pre_map(idd1(idd2),3)==map(cur_id(k),3));
                                    idd = [idd;idd3];
                                end;
                                if isempty(idd)
                                    repeats_c=[repeats_c;map(i,2)];
                                end;
                            end;
                        end;
                    end;
                end;
            else
            	cur_id = find(map(:,2) == map(i,2));
            	if ~isempty(cur_id)&&size(cur_id,1)~=1
                	idswitchflag_c = 1;%new target with a current repeat ID.
                	repeats_c=[repeats_c;map(i,2)];
            	end;
            end;
            if idswitchflag_s == 1
                mismatch_s_tmp = mismatch_s_tmp + 1;
            end;
            if idswitchflag_c == 1
                mismatch_c_tmp = mismatch_c_tmp + 1;
            end;
            if trueposflag_s == 1 
                truepos_s_tmp = truepos_s_tmp + 1;
            else
                truepos_c_tmp = truepos_c_tmp + 1;
            end;
        end;
        %
        repnum_s=unique(repeats_s,'rows');
        repnum_c=unique(repeats_c,'rows');
        mismatch_s_tmp = mismatch_s_tmp - size(repnum_s,1);
        mismatch_c_tmp = mismatch_c_tmp - size(repnum_c,1);
        %
    else
        map = [];
        mismatch_s_tmp = 0;
        mismatch_c_tmp = 0;
        truepos_s_tmp = 0;
        truepos_c_tmp = 0;
    end;
    
    %% Update parameters
    falsepos = falsepos + falsepos_tmp;
    missing = missing + missing_tmp;
    hypothesis = hypothesis + hypothesis_tmp;
    groundtruthes = groundtruthes + groundtruthes_tmp;
    mismatch_s = mismatch_s + mismatch_s_tmp;
    mismatch_c = mismatch_c + mismatch_c_tmp;
    truepos_s = truepos_s + truepos_s_tmp;
    truepos_c = truepos_c + truepos_c_tmp;
    
    %%  map to pre-map
    if isempty(map)
        pre_map = pre_map; 
    else if isempty(pre_map)
        pre_map = map;
        end;
    end;
    if ~isempty(map) && ~isempty(pre_map)
        %first step, find the intersection of map and pre_map
        %second step, update the intersection
        [C I_PRE I_MAP] = intersect(pre_map(:,1),map(:,1));
        pre_map(I_PRE,:) = [];
        pre_map = [pre_map; map];
    end;
end;
%% out put the parameter

precision = 1 - falsepos/hypothesis;
recall = 1 - missing/groundtruthes;

evalPerformance.gt = groundtruthes;
evalPerformance.results = hypothesis;
evalPerformance.missing = missing; 
evalPerformance.falsepos = falsepos; 
evalPerformance.mismatch_s = mismatch_s; 
evalPerformance.mismatch_c = mismatch_c;
evalPerformance.truepos_s = truepos_s; 
evalPerformance.truepos_c = truepos_c; 
evalPerformance.precision = precision; 
evalPerformance.recall = recall;
evalPerformance.mcta = (2*precision*recall/(precision+recall))*(1-mismatch_c/truepos_c)*(1-mismatch_s/truepos_s);


