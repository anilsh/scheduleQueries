function get_num_frames_q_learning_db3_icaps

% load dataset file
%load('./results_icaps/q_results_db3_icaps_test_final.mat', 'reward','rew_allep','results_sel');
%load('./results_icaps/q_results_db3_icaps_test_fn20_final.mat', 'reward','rew_allep','results_sel');
load('../results_icaps/q_results_db3_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

%load('./results_icaps/results_pICAPS_db3_test_exp2_traj_randSteps_3rew_nearTr_allRand.mat', 'reward','rew_allep','results_sel','pred_allP');
%load('./results_icaps/results_pICAPS_db3_test_exp2_traj_selectNearTr_70K.mat', 'reward','rew_allep','results_sel','pred_allP');
%load('./results_icaps/results_db3_test_exp2_traj_te1_tel400_selectNearTr_el200_allRand_f1.mat', 'reward','rew_allep','results_sel','pred_allP');

Cx = 5;
N = 4;
d = 2;
%% Time Gap For Inter-Camera Tracking (only #frames)

acc_p_r_EX_both = [];
acc_p_r_NN_both = [];
acc_p_r_EX_ict = [];
acc_p_r_NN_ict = [];

acc_p_r_ON = [];
%nnfPI = zeros(length(results_sel),2);
numFramesPi = zeros(length(results_sel),3);
numFramesEX = zeros(length(results_sel),3);
numFramesNN = zeros(length(results_sel),3);
numFramesON = zeros(length(results_sel),3);

% fill number of frames for online-learning approach
numFramesON(:,1) = [2702,2847,2203,375,2122,65,2796];
numFramesON(:,2) = [2953.0, 3009.0, 2814.0, 713.0, 3069.0, 82.0, 3062.0];
%numFramesON(:,3) = [-1,-1,-1,-1,-1,-1,-1];
numFramesON(:,3) = [14,19,12,0,4,1,17];    % numtransitions captured by Gaussian approach
numFramesON(:,4) = [2346,2822,1059,327,984,0,2246];   % missed + extra processed frames
%numFramesON(:,4) = [9.0, 1.0, 15.0, 2.0, 20.0, 48.0, 1.0];  % sum TR for all target individuals
%numFramesON(:,5) = [0.0, 0.0, 0.0, 0, 0.0, 0, 0.0];  % missed frames

acc = [];
P = [];
R = [];
C = zeros(N+1,N+1);
% For all person
for p = 1:length(results_sel)
    
    pid = results_sel{p};
    if isempty(pid)
        continue;
    end
    pid.g = single(pid.g);
    
    acc = [acc; pid.acc];
    P = [P; pid.prec];
    R = [R; pid.recall];
    confm = confusionmat(pid.g,pid.p);
    C = C + confm;
    %C = confusion(pid.g,pid.p);

    pr = pid.p;
    gt = pid.g;
    
    % compute time and numFrames for the policy
    %nnfPi(p,:) = sum(gt==Cx & pr~=Cx) + sum(pr~=gt & gt~=Cx & pr~=Cx);
    numFramesPi(p,:) = compute_num_frames_1_person(pid.p,pid.g, Cx);
    
    % compute time and numFrames for the exhaustive search
    numFramesEX(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'EX');
    
    % compute time and numFrames for the nearest neighbor
    numFramesNN(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'NN');
    
    % compute accuracy for exhaustive and NN methods
    acc_p_r_EX_both = [acc_p_r_EX_both; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*N),1.0];
    acc_p_r_NN_both = [acc_p_r_NN_both; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*(d+1)),1.0];
    acc_p_r_EX_ict = [acc_p_r_EX_ict; numFramesEX(p,3)/numFramesEX(p,1), numFramesEX(p,3)/numFramesEX(p,2),1.0];
    acc_p_r_NN_ict = [acc_p_r_NN_ict; numFramesNN(p,3)/numFramesNN(p,1), numFramesNN(p,3)/numFramesNN(p,2),1.0];
    
    acc_p_r_ON = [acc_p_r_ON; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx),1.0];
    
    
end

% consider people having transitions in multiple cameras
nfPI = numFramesPi;
nfPI(nfPI(:,1)==0,:) = [];
nfEX = numFramesEX;
nfEX(nfEX(:,1)==0,:) = [];
nfNN = numFramesNN;
nfNN(nfNN(:,1)==0,:) = [];
nfON = numFramesON;
nfON(nfON(:,1)==0,:) = [];
% nfON(nfON(:,2)==0,2) = 1;

% nfEX = nfEX(1:4,:);
% nfNN = nfNN(1:4,:);
% nfON = nfON(1:4,:);
% nfPI = nfPI(1:4,:);


%nn = numFramesPi;
%nn(nn(:,1)==0,:) = [];
disp( mean([nfPI,nfEX,nfNN]) );  % 

% accuracy, precision and recall of other (exhaustive and neighbor) approach for ICT 
disp( mean(nfEX(:,3)./nfEX(:,1)) );
disp( mean(nfEX(:,3)./nfEX(:,2)) );
disp(1);

disp( mean(nfNN(:,3)./nfNN(:,1)) );
disp( mean(nfNN(:,3)./nfNN(:,2)) );
disp(1);

% accuracy, precision and recall of Gaussian approach for ICT 
%%%disp( mean(nfEX(:,3)./(nfON(:,1)-(nfON(:,2)/(d+1)))) );
%disp( mean((nfEX(:,3)+(nfON(:,4)-nfON(:,5)))./nfON(:,1) ));
disp( mean((nfON(:,1)-nfON(:,4))./nfON(:,1) ));
% (alltime-missed-extraProcess)/alltime
disp( mean(nfEX(:,3)./nfON(:,2)) );
% correct
%disp( mean((nfEX(:,3)-(nfON(:,5)~=0))./nfEX(:,3)) );
disp( mean((nfON(:,3))./nfEX(:,3)) );
% nfON(3)/nfEX(3)

% accuracy of pi for ICT
disp( mean((nfPI(:,1)-nfPI(:,2))./nfPI(:,1)) );
% precision of pi for ICT (total transitions)
disp( mean(nfPI(:,3)./nfPI(:,2)) );
% recall of pi for ICT
disp( mean(nfPI(:,3)./nfEX(:,3)) );

disp('');

% plot all required figures
close all; 
figure; hold on;
m = mean(nfEX); plot(m(1),m(2), '*r', 'MarkerSize',18);
m = mean(nfNN); plot(m(1),m(2), '+r', 'MarkerSize',18);
m = mean(nfON); plot(m(1),m(2), 'xr', 'MarkerSize',18);
m = mean(nfPI); plot(m(1),m(2), 'or', 'MarkerSize',18);
legend('Exhaustive','Neighbor','Gaussian','Policy');
xlabel('Time (#frames)');
ylabel('Number of Re-ID calls ');
box on;

figure; hold on; box on;
colorstring = 'rgbkymc';
m = nfEX; 
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(i),'*'),'MarkerSize',18)
end
m = nfNN; 
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(i),'+'),'MarkerSize',18)
end
m = nfON; 
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(i),'x'),'MarkerSize',18)
end
m = nfPI; 
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(i),'o'),'MarkerSize',18)
end

xlabel('Time (#frames)');
ylabel('Number of Re-ID calls ');


figure; box on;
boxplot([nfEX(:,2),nfNN(:,2),nfON(:,2),nfPI(:,2)],'Labels',{'Exhaustive','Neighbor','Gaussian','Policy'});
ylabel('Number of Re-ID calls ');

figure; box on;
boxplot([nfEX(:,1),nfNN(:,1),nfON(:,1),nfPI(:,1)],'Labels',{'Exhaustive','Neighbor','Gaussian','Policy'});
ylabel('Time taken to find the target');

figure; box on;
a = mean([nfEX,nfNN,nfON,nfPI]);
boxplot([nfEX(:,2),nfNN(:,2),nfON(:,2),nfPI(:,2)],'Labels',{num2str(a(1)),num2str(a(3)),num2str(a(5)),num2str(a(7))});
ylabel('Number of Re-ID calls ');
xlabel('Time (#frames)');

disp([mean(acc),mean(P),mean(R)]);
disp(C./sum(C,2));
disp('');



function val = compute_num_frames_1_person(p,g, Cx)

val = zeros(1,3);
count_ict = 0;
n = 0;
t = 0;
inc = 1; % indicator for inside camera
a = find(g ~= Cx);
for i = 1:a(end)
    
    if inc==1 && g(i)~=Cx % inside single camera
        % do nothing
    elseif inc==0 && g(i)==p(i) && g(i)~=Cx % entered in Ci (single camera)
        %disp('Entered in Ci: ');
        %disp([n,t]); 
        count_ict = count_ict+1;
        inc = 1;
        % do nothing as selected camera is Ci
    elseif inc==1 && g(i)==Cx  % entered into Cx
        %disp('Entered in Cx: ');
        %disp([n,t]); 
        
        inc = 0;
        % increments time-step and whether camera selected
        t = t+1;
        n = n + (p(i)~=Cx);
    else % still doing transition
        % increments time-step and whether camera selected
        t = t+1;
        n = n + (p(i)~=Cx);
    end
   
end

val(1) = t;
val(2) = n;
val(3) = count_ict;
disp('');

function val = compute_num_frames_ex_nn(g, Cx,N,d, task)

val = zeros(1,3);
count_ict  = 0;
n = 0;
t = 0;
a = find(g ~= Cx);
for i = 2:a(end)
    if g(i) == g(i-1) && g(i)~=Cx   % inside single camera
        continue;
    end
    
    t = t + 1;
    if task == 'EX'
        n = n + N;
    elseif task == 'NN'
        n = n + (d+1);
    end
    if g(i) ~= Cx && g(i-1)==Cx
        count_ict = count_ict+1;
    end
end

val(1) = t;
val(2) = n;
val(3) = count_ict;
disp('');

