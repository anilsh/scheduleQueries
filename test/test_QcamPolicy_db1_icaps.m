function  [tot_reward,rew_allep,results_sel,pred_allP] = test_QcamPolicy_db1_icaps(opts, Qc)
% Camera Selection, ICAPS 2019
% 
% Anil Sharma, IIIT-Delhi
%

figure;

ah_len = 4;  % action-history length
d = 8;       % size of region matrix (r) 
opts.d = d; 
opts.lte = 200; % maximum length of telapse

if nargin < 2
   disp('error. Q-table not provided.');
end

% load ground-truth trajectories
load('./data/ann_MCT_dataset1_pidWise.mat');

alltime = 24000;
fpsc = 1;    % to downsample the frames-per-sec
% following random sequence is used to create train-test set
pALL = [88 137 182 223 174 71 200 58 1 185 74 168 235 46 17 67 162 100, ...
    192 113 140 6 94 81 11 181 159 78 147 22 127 115 68 143 59 212 217 161,...
    92 164 227 206 73 123 125 126 230 233 83 197 231 34 145 26 114 27 138 7, ...
    151 8 48 120 166 215 35 85 64  111 156 225 65 171 153 25 45 207 219 178,...
    23 165 50 199 110 203 211 47 152 30 132 102 205 96 89 129 52 107 60 36,...
    216 167 87 149 208 116 213 57 41 214 12 175 163 15 173 144 134 86 194 82,...
    128 186 63 105 122 69 21 183 169 187 222 19 232 108 198 79 141 91 51 150,...
    53 77 5 119 39 33 170 84 229 180 133 40 188 139 54 121 158 55 42 10 9 124,...
    142 90 136 189 226 131 135 13 95 24 209 191 44 29 16 218 20 93 184 130 117,...
    49   204     3   112   146   195   109    62    37    98   154    72    99,...
    157  234   103    56   106    31    66   172   202 38 43 28 101 221 193 177,...
    2 4 160 155 14 210 196 176 97 70 201 228 75 18 104 179 118 148 190 224 220,...
    32 61 80 76];

% initialize variables to stores results
pred_allP = [];
rew_allep = [];
results_sel = [];
rs_cnt = 1;
MCTA = [];
tot_reward = [];
tic_train = tic;
for epoch = 1 : 1 
    % repeat for all pedestrians
    %disp(pALL);
    rew_thisep = 0;
    for p = pALL(2:2:end) 
        pred_allP{p} = [];
        pr_cam_allt = [];
        gt_cam_allt = [];
        ped = PID{p};
        
        % testing sequence starts from start to end of target's trajectory
        startIDX = 1; % index_of_randCam(randi(len_indices_of_randCam)); %randi( min(100,size(ped,1)-100) );
        myPos = ped(startIDX,1:2);
        curr_camera = myPos(1);
        curr_frame = myPos(2);
        next_cam_gt = curr_camera;
        
        numSwitches = 0;
        numAccurate = 0;
        numAccSwitches = 0;
        
        % make history vector
        history.c = curr_camera*ones(ah_len, 1);
        history.te = 1;
        telapse = 0;
        num_steps = 0;
        
        numImages = 0;
        %prev_score = -inf;

        bbox_pt = ped(ped(:,1)==curr_camera&ped(:,2)==curr_frame,3:6);
        isPresent = 1;
        pred_allP{p} = [pred_allP{p}; [curr_camera,curr_frame,p,bbox_pt]];
        
            
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

            % go to next frame after camera selection
            curr_frame = curr_frame+fpsc;

            % find the ground truth camera
            next_cam_gt = findTarget(ped,curr_frame,opts);
            
            pr_cam_allt = [pr_cam_allt; next_camera];
            gt_cam_allt = [gt_cam_allt; next_cam_gt];

            if next_cam_gt == next_camera
                numAccurate = numAccurate + 1;
            end
            
            %tic;
            %[isPresent,bbox_pt_n] = isTargetPresent(p,next_camera,curr_frame, Hist_test,testID,featAllBbox,featDetails);
            isPresent = isTargetPresent_gt(ped,next_camera,curr_frame);
            bbox_pt_n = ped(ped(:,1)==next_camera&ped(:,2)==curr_frame,3:6);
            
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