%function [Qc,MCTA,results_det,tot_reward] = train_QcamPolicy_db3_aamas(opts, Qc)
function [Qc,tot_reward,rew_allep] = train_db3_postICAPS_3rew_selectNearTr_te2(opts, Q_table_name, Qc)
% Camera Selection, AAMAS 2019
% 
% Anil Sharma, IIIT-Delhi
%
opts.alwaysPresent = 0;

ah_len = 4;
d = 8;
opts.d = d;
opts.lte = 400; % length of telapse
if nargin < 3
   Qc = (-1*ones(opts.num_camera,d*d,(opts.num_camera+1)^(ah_len),opts.lte,opts.num_camera+1,'single'));
   
end

episode_length = 100;
% load ground-truth trajectories
load('./ann_MCT_dataset3_pidWise.mat');

alltime = 5251;
fpsc = 2;
%pALL = randperm(14);
pALL = [9     1     5     2     8     7     6    12    13    11     4    14    10     3];

rew_allep = [];
results_det = [];
MCTA = [];
tot_reward = [];
tic_train = tic;
for epoch = 50001 : opts.numEpoch
    % repeat for all pedestrians
    disp(pALL);
    rew_thisep = 0;
    for p = pALL(1:2:end) 
        pr_cam_allt = [];
        gt_cam_allt = [];
        ped = PID{p};
        %ped(:,1) = ped(:,1)+1;
        
        % to select start index from all camera uniquely
        %uniq_cam = unique(ped(:,1));
        %randCam = uniq_cam(randi(length(uniq_cam)));
        %index_of_randCam = find( ped(ped(:,1)==randCam,1) );
        %len_indices_of_randCam = length(index_of_randCam);
        
        tranIDX = find(ped(2:end,1) - ped(1:end-1,1));
        if rand < 0.4 % select anywhere
            startIDX = randi( length(ped(:,1))-50 ); %index_of_randCam(randi(len_indices_of_randCam));
        else % select near transition
            startIDX  = tranIDX(randi( length(tranIDX) ))-10;
        end
        
        
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
        
        totTimeMiss = 0;
        numImages = 0;
        %prev_score = -inf;

        pos_examples = []; pos_labels = []; 
        neg_examples = []; neg_labels = []; 
        
        
        bbox_pt = ped(ped(:,1)==curr_camera&ped(:,2)==curr_frame,3:6);
        isPresent = 1;
        
            
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
                %pos_bbox = single(gen_samples('gaussian', curr_bbox_gt, 100, opts, opts.finetune_trans, opts.finetune_scale_factor));
                %r = overlap_ratio(pos_bbox,curr_bbox_gt);
                %pos_bbox = pos_bbox(r>0.9,:);
                %pos_bbox = pos_bbox(...
                %      pos_bbox(:,1)>1 & pos_bbox(:,2)>1 & pos_bbox(:,3)>1 & pos_bbox(:,4)>1,:);

            else
                %curr_bbox = [1 1 1 1];

            end
            %telapse = telapse+1;

            
            if isempty(pos_bbox)
                pos_bbox = curr_bbox_gt;
                disp('');
            end
            curr_bbox = pos_bbox(randi(size(pos_bbox,1)),:);
            
            % Select a camera using Q-learning
            [next_camera,rtA,chistn,teH] = find_next_camera_usingQc(Qc,history,curr_camera,curr_bbox, opts);

            % epsilon-greedy exploration
            epsilon = 1/ log(epoch);
            if rand < epsilon
                % take random action
                next_camera = randi(opts.num_camera+1);
            else
                % use the selected camera
            end
            
            % take random steps
            if rand < -0.4
                r = randi(50);
            else
                r = 1;
            end
            telapse = telapse+r;
            curr_frame = curr_frame+r;
            %curr_frame = curr_frame+fpsc;
            % read next image  here
            if curr_frame >= alltime
                break;
            end

            % find the ground truth camera
            next_cam_gt = findTarget(ped,curr_frame,opts);
            
            pr_cam_allt = [pr_cam_allt; next_camera];
            gt_cam_allt = [gt_cam_allt; next_cam_gt];

            if next_cam_gt == next_camera && next_camera ~= opts.num_camera+1
                numAccurate = numAccurate + 1;
                rew_thisep = rew_thisep+1;
            elseif next_cam_gt == next_camera && next_camera == opts.num_camera+1
                numAccurate = numAccurate + 1;
                rew_thisep = rew_thisep+0;
            else
                rew_thisep = rew_thisep-1;
            end
            

            %tic;
            %[isPresent,bbox_pt_n] = isTargetPresent(p,next_camera,curr_frame, Hist_test,testID,featAllBbox,featDetails);
            isPresent = isTargetPresent_gt(ped,next_camera,curr_frame);
            bbox_pt_n = ped(ped(:,1)==next_camera&ped(:,2)==curr_frame,3:6);
            
            % find labels (pos if target present in next camera)
            %thresReached = -1;
            for rti = 1:length(rtA)
                rt = rtA(rti);
                
                %for c = 1: opts.num_camera
                if next_camera ~= opts.num_camera+1
                    if isTargetPresent_gt(ped,next_camera,curr_frame)
                        pos_examples = cat(1, pos_examples, [curr_camera,rt,chistn,1,teH]);
                        pos_labels = cat(1,pos_labels,next_camera);
                    else
                        neg_examples = cat(1, neg_examples, [curr_camera,rt,chistn,0,teH]);
                        neg_labels = cat(1,neg_labels,next_camera);
                    end
                %end
                else
                    if ~isTargetVisible_gt(ped,curr_frame,opts)
                        pos_examples = cat(1, pos_examples, [curr_camera,rt,chistn,1,teH]);
                        pos_labels = cat(1,pos_labels,opts.num_camera+1);
                    else
                        neg_examples = cat(1, neg_examples, [curr_camera,rt,chistn,0,teH]);
                        neg_labels = cat(1,neg_labels,opts.num_camera+1);
                    end
                end
                
            end
            
            % update current location of the target
            if size(bbox_pt,1) > 1
                disp('');
            end
            if ~isempty(bbox_pt_n)
                bbox_pt = bbox_pt_n;
                curr_camera = next_camera;
            end

            numImages = numImages+1;
            num_steps = num_steps+1;
            %prev_score = curr_score;

            % update history vector
            if ~isPresent
                %telapse = telapse +1;
                history.te = min(ceil(telapse/2), opts.lte);
            else
                telapse = 0;
                history.te = 1;
            end
            history.c(2:end) = history.c(1:end-1);
            history.c(1) = next_camera;
            
            if num_steps > episode_length && next_cam_gt~=opts.num_camera+1
                break;
            end
            
        end
            
        % update the Q-table using Q-learning
        if ~isempty(pos_examples)
            Qc = train_Qc(Qc, pos_examples,pos_labels, +1,opts);
        end
        if ~isempty(neg_examples)
            Qc = train_Qc(Qc, neg_examples,neg_labels, -1,opts);
        end
        
        
        % Print objective (loss)
        fprintf('time: %04.2f, epoch: %02d/%02d, curr_frame: %d, (cam,ped): (%d,%d) \n', ...
                toc(tic_train), epoch, opts.numEpoch, curr_frame,curr_camera,p);
        fprintf('numSwicthes:%d, numAccurate:%d, numAccSwicthes:%d, numImages:%d, ratio:%f \n', ...
                numSwitches, numAccurate, numAccSwitches, num_steps,numAccurate/length(pr_cam_allt));
            
        % training over positive and negative samples of current batch has finished.
        %figure; plot(pr_cam_allt)
        %hold on; plot(gt_cam_allt-0.1)
        
        %mcta = compute_mcta(pr_cam_allt,gt_cam_allt, opts);
        %results_det{p}.g = gt_cam_allt;
        %results_det{p}.p = pr_cam_allt;
        %results_det{p}.o = opts;
        
        %MCTA = [MCTA; mcta];
        disp('');
        rew_allep = [rew_allep; rew_thisep]; 
    end
    
    if mod(epoch,100) == 0
        % save Q-table
        save([Q_table_name,'_tmp'], 'Qc','-v7.3')
    end
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


function Qc = train_Qc(Qc, samples,actions, reward, opts)
alpha = 0.2;
gamma = 0.8;
c = samples(:,1);
rt = samples(:,2);
chist = samples(:,3);
phist = samples(:,4);
teC = samples(:,5);

batch_size = 1;
for i = 1:ceil(length(chist)/batch_size)
    batch_idx = (i-1)*batch_size+1: i*batch_size;
    if ( batch_idx(end) >= length(chist) )
        batch_idx = (i-1)*batch_size+1: length(chist);
    end
    
    % get current q-value
    Qval = single(Qc(c(batch_idx),rt(batch_idx),chist(batch_idx),teC(batch_idx),actions(batch_idx)));
    % get next Q-value at next state
    Qtplus1 = find_qtplus1(Qc, c(batch_idx),chist(batch_idx),phist(batch_idx),teC(batch_idx),actions(batch_idx), opts);
    
    %Qval = (1-alpha) * Qval + alpha*(reward + gamma*Qtplus1);
    if actions(batch_idx) == opts.num_camera+1 && reward == 1
        Qval = (1-alpha) * Qval + alpha*(0 + gamma*Qtplus1);
    else
        Qval = (1-alpha) * Qval + alpha*(reward + gamma*Qtplus1);
    end
    
    %Qc(rt(batch_idx),chist(batch_idx),phist(batch_idx),te1(batch_idx),te2(batch_idx),te3(batch_idx),actions(batch_idx)) = Qval;
    Qc(c(batch_idx),rt(batch_idx),chist(batch_idx),teC(batch_idx),actions(batch_idx)) = Qval;

end


function Qval = find_qtplus1(Qc, c,ch,p,teC, actions,opts)

Qval = zeros(size(ch,1),1);
for i = 1: size(ch,1)
    %[phi,phist] = num2phist(ph(i));
    
    % update camera history with selected action
    chi = num2chist(ch(i),actions(i));
    % increment telapse, if not present
    if p == 1 && (actions(i) ~= opts.num_camera+1)
        teC = 1;
        % change (c,r)
        c = actions(i);
        
    else 
        teC = min(teC+1,opts.lte);
    end

    %disp([c,chi,teC]);
    Qval(i) = max(max( Qc(c,:,chi,teC,:) ));
    %Qval(i) = max(max( Qc(:,chi,phi,tei,:) ));
end

function num = num2chist(chnum,action)
ah_len = 4;
c = dec2base(chnum-1,5);
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
    numC = 4;
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