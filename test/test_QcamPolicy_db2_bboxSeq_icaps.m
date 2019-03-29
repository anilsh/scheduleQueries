function  [tot_reward,rew_allep,results_sel,pred_allP] = test_QcamPolicy_db2_bboxSeq_icaps(opts, Qc)
% Camera Selection, ICAPS 2019
% 
% Anil Sharma, IIIT-Delhi
%
figure;

ah_len = 4; % action-history length
d = 8;      % divide image into dxd region
opts.d = d;
opts.lte = 200; % maximum length of telapse
if nargin < 2
   disp('error. Q-table not provided.');
   
end

% load ground-truth trajectories
load('./data/ann_MCT_dataset2_pidWise.mat');

alltime = 24000;
fpsc = 1;
pALL = 1:255; %randperm(255);   % all-id's are used

pred_allP = [];
rew_allep = [];
results_sel = [];
rs_cnt = 1;
MCTA = [];
tot_reward = [];
tic_train = tic;
for epoch = 1 : 1 
    % repeat for all pedestrians
    disp(pALL);
    rew_thisep = 0;
    for p = pALL(1:1:end) 
        pred_allP{p} = [];
        pr_cam_allt = [];
        gt_cam_allt = [];
        ped = PID{p};
        %ped(:,1) = ped(:,1)+1;
        
        startIDX = 1; %index_of_randCam(randi(len_indices_of_randCam)); %randi( min(100,size(ped,1)-100) );
        myPos = ped(startIDX,1:2);
        curr_camera = myPos(1);
        curr_frame = myPos(2);
        next_cam_gt = curr_camera;
        
        numSwitches = 0;
        numAccurate = 0;
        numAccSwitches = 0;
        
        % make history vector
        history.c = curr_camera*ones(ah_len, 1);
        %history.p = 1;
        history.te = 1;
        telapse = 0;
        num_steps = 0;
        inc = 1;
        
        totTimeMiss = 0;
        numImages = 0;
        %prev_score = -inf;
        
        
        bbox_pt = ped(ped(:,1)==curr_camera&ped(:,2)==curr_frame,3:6);
        isPresent = 1;
        pred_allP{p} = [pred_allP{p}; [curr_camera,curr_frame,p,bbox_pt]];
        
        % store previous bbox
        pre_bbox_sel = bbox_pt;
        pre_bbox_polled = bbox_pt;
        match_count = 1;
            
        while(curr_frame <= ped(end,2) )
            % read next image  here
            if curr_frame >= alltime
                break;
            end

            
            if numImages == 0 || isPresent
                
                opts.imgSize = [240 320 3];
                %curr_bbox = ped(curr_frame,3:6); %vid_info.gt(nextFrame,:);
                curr_bbox_gt = bbox_pt; %bbox_pt = ped(ped(:,1)==curr_camera&ped(:,2)==curr_frame,3:6);
                pos_bbox = [];
                
            end
            
            if isempty(pos_bbox)
                pos_bbox = curr_bbox_gt;
                disp('');
            end
            curr_bbox = pos_bbox(randi(size(pos_bbox,1)),:);
            
            % Select a camera using Q-learning
            [next_camera,~,~,~] = find_next_camera_usingQc(Qc,history,curr_camera,curr_bbox, opts);
            
            curr_frame = curr_frame+fpsc;

            % find the ground truth camera
            next_cam_gt = findTarget(ped,curr_frame,opts);
            
            % Select SCT or ICT+SCT
            req_only_ict = 1;
            if req_only_ict == 1
                Cx = opts.num_camera+1;
                if inc==1 && next_cam_gt~=Cx % inside single camera
                    next_camera = next_cam_gt;
                    % do nothing
                elseif inc==0 && next_cam_gt==next_camera && next_cam_gt~=Cx % entered in Ci (single camera)
                    %disp('Entered in Ci: ');
                    %disp([n,t]); 

                    inc = 1;
                    % correct selection by policy, use ground truth
                    next_camera = next_cam_gt;
                elseif inc==1 && next_cam_gt==Cx  % entered into Cx
                    %disp('Entered in Cx: ');
                    %disp([n,t]); 

                    inc = 0;
                    % do nothing, use selected camera
                    
                else % still doing transition
                    % do nothing, use selected camera
                    % 
                end
            end
            
            pr_cam_allt = [pr_cam_allt; next_camera];
            gt_cam_allt = [gt_cam_allt; next_cam_gt];

            if next_cam_gt == next_camera
                numAccurate = numAccurate + 1;
            end
            

            %tic;
            %[isPresent,bbox_pt_n] = isTargetPresent(p,next_camera,curr_frame, Hist_test,testID,featAllBbox,featDetails);
            isPresent = isTargetPresent_gt(ped,next_camera,curr_frame);
            bbox_pt_n = ped(ped(:,1)==next_camera&ped(:,2)==curr_frame,3:6);
            
            % simulate Re-ID errors
            % simulate errors in Re-ID
            if rand < 0.2  && (inc == 0 || req_only_ict==0) % flip presence information
%                 if isPresent == 1
%                     isPresent = 0;
%                     bbox_pt_n = [];
%                 else
                    if next_camera == opts.num_camera+1
                        % do nothing, always not present
                        isPresent = 0;
                        bbox_pt_n = [];
                    else
                        isPresent = 1;
                        % provide a random bbox (not having target) from the curr_frame 
                        % read selected camera gt file
                        fpath = opts.fpath;
                        gt = csvread([fpath,'/cam',num2str(next_camera),'.dat']);
                        % find all bbox and pid at curr_frame
                        all_pid_curr_frame = gt(gt(:,2)==curr_frame,3);
                        all_bboxes_curr_frame = gt(gt(:,2)==curr_frame,4:7);
                        alter_pid = find(all_pid_curr_frame~=p);
                        % select bbox without pid==p
                        if isempty(alter_pid)
                            % do nothing, send not present information
                            isPresent = 0;
                            bbox_pt_n = [];
                        else
                            bbox_pt_n = all_bboxes_curr_frame(randi(length(alter_pid)),:);
                        end
                    end
                    
                %end
                
            else  % use as is
                
            end
            
%             if all(history.c == opts.num_camera+1)
%                 % clear prev_bbox
%                 pre_bbox_polled = [];
%             end
%             if isempty(pre_bbox_polled) && ~isempty(bbox_pt_n)
%                 pre_bbox_polled = bbox_pt_n;
%                 match_count = 0;
%             end
            if ~isempty(bbox_pt_n)
                iou_polled = bboxOverlapRatio(bbox_pt_n,pre_bbox_polled);
                iou_sel = bboxOverlapRatio(bbox_pt_n,pre_bbox_sel);
                if iou_polled < 0.6 && iou_sel < 0.6 % dont use this bbox, rather update polled
                    pre_bbox_polled = bbox_pt_n;
                    
                    match_count = 0;
                else
                    % use current bbox and update pre_bbox
                    pre_bbox_sel = bbox_pt_n;
                    pre_bbox_polled = bbox_pt_n;
                    match_count = match_count + 1;
                end
            else
                match_count = 0;
 
            end
            
            % select cuurent bbox if IOU matching with previous
            if match_count >= 1
                % use selected bbox
                isPresent = 1;
            else
                bbox_pt_n = [];
                isPresent = 0;
            end
            
            % update current location of the target
            if size(bbox_pt,1) > 1
                disp('');
            end
            if ~isempty(bbox_pt_n)
                bbox_pt = bbox_pt_n;
                curr_camera = next_camera;
                rew_thisep = rew_thisep+1;
            else
                rew_thisep = rew_thisep-1;
            end

            numImages = numImages+1;
            num_steps = num_steps+1;
            %prev_score = curr_score;

            % update history vector
            if ~isPresent
                telapse = telapse +1;
                history.te = min(ceil(telapse/2), opts.lte);
            else
                telapse = 0;
                history.te = 1;
            end
            history.c(2:end) = history.c(1:end-1);
            history.c(1) = next_camera;
            
            if isPresent %isTargetPresent(ped,next_camera,curr_frame)
                pred_allP{p} = [pred_allP{p}; [curr_camera,curr_frame,p,bbox_pt]];
            else
                %pred_allP{p} = [pred_allP{p}; [0,0,0,0,0,0]];
            end
            
        end
            
                
%         subplot(length(pALL(2:2:end)),1,rs_cnt);
%         a= [(repmat(gt_cam_allt',100,1)); zeros(5,length(gt_cam_allt)); (repmat(pr_cam_allt',100,1))];
%         colormap('hot');
%         imagesc(a);
%         set(gca,'ytick',50:100:200)
%         set(gca,'yticklabel',{'GT','Sel'})
        
        % Print objective (loss)
        fprintf('time: %04.2f, epoch: %02d/%02d, curr_frame: %d, (cam,ped): (%d,%d) \n', ...
                toc(tic_train), epoch, opts.numEpoch, curr_frame,curr_camera,p);
        fprintf('numSwicthes:%d, numAccurate:%d, numAccSwicthes:%d, numImages:%d, ratio:%f \n', ...
                numSwitches, numAccurate, numAccSwitches, num_steps,numAccurate/length(pr_cam_allt));
            
        % training over positive and negative samples of current batch has finished.
        %figure; plot(pr_cam_allt)
        %hold on; plot(gt_cam_allt-0.1)
        
        results_sel{rs_cnt}.g = gt_cam_allt;
        results_sel{rs_cnt}.p = pr_cam_allt;
        results_sel{rs_cnt}.acc = sum(pr_cam_allt==gt_cam_allt)/length(gt_cam_allt);
        results_sel{rs_cnt}.prec = sum(pr_cam_allt(pr_cam_allt~=opts.num_camera+1) ...
                                    ==gt_cam_allt(pr_cam_allt~=opts.num_camera+1)) ...
                                    /length(gt_cam_allt(pr_cam_allt~=opts.num_camera+1));
        results_sel{rs_cnt}.recall = sum(pr_cam_allt(gt_cam_allt~=opts.num_camera+1) ...
                                    ==gt_cam_allt(gt_cam_allt~=opts.num_camera+1)) ...
                                    /length(gt_cam_allt(gt_cam_allt~=opts.num_camera+1));
        rs_cnt = rs_cnt+1;
        
        %MCTA = [MCTA; mcta];
        disp('');
        
    end
    
    rew_allep = [rew_allep; rew_thisep];
    
   
end


function img = imread_multicam(cam,framenum)
dataset_path = '/media/win/data/MCT dataset/dataset3/';

cam_path = [dataset_path,'cam',num2str(cam),'/'];
%imgpath = fullfile(cam_path,sprintf('%04d.png',framenum));
imgpath = fullfile(cam_path,sprintf('%04d.png',framenum));

img = imread(imgpath);

function [rt,r1,rx1,ry1,r2,rx2,ry2] = find_curr_rt(bbox,opts)

if bbox(3) == 1 && bbox(4) == 1
    %rt = [1,1];
    rt = 1;
    r1 = 1; rx1 = 1; ry1 = 1;
    r2 = 1; rx2 = 1; ry2 = 1;
else
    [r1,rx1,ry1] = get_region_num(bbox(2),bbox(1),opts);
    [r2,rx2,ry2] = get_region_num(bbox(2)+bbox(4),bbox(1)+bbox(3),opts);
    %[r1,rx1,ry1] = get_region_num(bbox(2)+bbox(4)/2,bbox(1)+bbox(3)/2,opts);
    %[r2,rx2,ry2] = get_region_num(bbox(2)+bbox(4)/2,bbox(1)+bbox(3)/2,opts);
    rt = r1; %[r1,r2];
end

function [r,rx,ry] = get_region_num(x,y,opts)

d = opts.d;
cell_size = opts.imgSize/d;

% find regions for x and y direction
rx = ceil( x / cell_size(2) );
ry = ceil( y / cell_size(1) );
if (rx <=0 || ry <= 0)
    disp('');
end
r = d*(rx-1) + ry;



function num = num2chist(chnum,action)
ah_len = 4;
c = dec2base(chnum-1,4);
chi = zeros(ah_len,1);
chi(end:-1:end-length(c)+1) = c(end:-1:1)-48;
chi(2:end) = chi(1:end-1)+1;
chi(1) = action;
num = get_cam_history_num(chi)+1; 

% % check if the history will remain same
% if chi(2) == action && p == 1
%     % remain same
%     num = chnum;
%     %teH = te+1;
% else
%     
%     chi(1) = action;
%     num = get_cam_history_num(chi)+1; 
%     %teH = 1;
% end


function [num,phi] = num2phist(phnum)
ah_len = 4;
p = dec2base(phnum-1,2);
phi = zeros(ah_len,1);
phi(end:-1:end-length(p)+1) = p(end:-1:1)-48;
phi(2:end) = phi(1:end-1);
phi(1) = 1;
num = get_presence_history_num(phi)+1;

function num = get_cam_history_num(ch,opts)
if nargin < 2
    numC = 3;
else
    numC = opts.num_camera;
end
num = 0;
for i = 1:length(ch)
    v = ch(end-i+1);
    if v == 0
        disp('');
    else
        num = num + (v-1)*((numC+1)^(i-1));
    end    
end

function num = get_TE_history_num(te_h,opts)
maxTE = opts.nte;

num = 0;
for i = 1:length(te_h)
    v = te_h(end-i+1);
    if v == 0
        disp('');
    else
        num = num + (v-1)*((maxTE)^(i-1));
    end
end

function num = get_presence_history_num(ch)
num = 0;
for i = 1:length(ch)
    num = num + ch(end-i+1)*(2^(i-1));
end

function [next_camera,rt,cam_hist,te] = find_next_camera_usingQc(Qc,history,c,curr_bbox, opts)
d = opts.d;
% find r_t using curr_bbox
[~,~,rx1,ry1,~,rx2,ry2] = find_curr_rt(curr_bbox,opts);
cam_hist = get_cam_history_num(history.c,opts)+1;
%p_hist = get_presence_history_num(history.p)+1;
te = history.te;

next_cam_list = [];
for rx = rx1:rx2
    for ry = ry1:ry2
        rt = d*(rx-1) + ry;
        
        if all(Qc(c,rt,cam_hist,te,1) == Qc(c,rt,cam_hist,te,:))
        %if all(Qc(rt,cam_hist,p_hist,te_hist,1) == Qc(rt,cam_hist,p_hist,te_hist,:))
            next_camera = randi(opts.num_camera+1);
            rval = -1;
            %next_camera = history.c(1);
        else
            [rval,next_camera] = max (Qc(c,rt,cam_hist,te,:) );
            %[rval,next_camera] = max (Qc(rt,cam_hist,p_hist,te_hist,:) );
            
        end
        
        next_cam_list = [next_cam_list; next_camera,rt,rval];
        
    end
end
%next_camera = mode(next_cam_list(:,1));
a = unique(next_cam_list(:,1));
out = [a,histc(next_cam_list(:,1),a)];
% maximum occuring number
maxV = max(out(:,2));
maxIdx = find(out(:,2) == maxV);
if length(maxIdx) == 1
    next_camera = out(maxIdx,1);
else
    % select a random of all voted, if same number of cells voting
    next_camera = out(maxIdx(randi(length(maxIdx))),1);
end
%if any(out(maxIdx,1) == history.c(1))
%    next_camera = history.c(1);
%else
%    if( length(maxIdx) > 1)
%        disp('');
%    end
%    next_camera = out(maxIdx(1),1);
%end
if isempty(next_camera)
    disp('qqqwerty');
end
rt = next_cam_list(:,2);
% find next camera using Qc or using random selection
%disp([rt(1),rt(2),cam_hist,p_hist]);
%[~,next_camera] = max (Qc(rt(1),rt(2),cam_hist,p_hist,:) );


function myc = findTarget(ped,frameNum,opts)
myc = 0;
for c = 1:opts.num_camera
    if ~isempty( find(ped(ped(:,1)==c & ped(:,2) == frameNum,:)) )
        myc = c;
        break;
    end
end
if myc == 0
    myc = opts.num_camera+1;
end

% function myc = findTarget_onlyCam(ped,frameNum,opts)
% 
% cur_index = find(ped(:,2) > frameNum);
% if isempty(cur_index)
%     myc = opts.num_camera+1;
% else
%     myc = ped(cur_index(1),1);
% end


function p = isTargetPresent_gt(ped,camNum,frameNum)

if ~isempty( find(ped(ped(:,1)==camNum & ped(:,2) == frameNum,:)) )
    p = 1;
else
    p = 0;
end

function p = isTargetVisible_gt(ped,frameNum,opts)
% ped(:,1) = ped(:,1)-1;
for c = 1:opts.num_camera
    if ~isempty( find(ped(ped(:,1)==c & ped(:,2) == frameNum,:)) )
        p = 1;
        break;
    else
        p = 0;
    end
end