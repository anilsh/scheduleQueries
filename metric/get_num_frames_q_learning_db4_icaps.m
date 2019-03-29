function get_num_frames_q_learning_db4_icaps

% load dataset file
%load('./results/q_results_db4_icaps_test.mat', 'reward','rew_allep','results_sel');
%load('./results_icaps/q_results_db4_icaps_test_final.mat', 'reward','rew_allep','results_sel');
load('../results_icaps/q_results_db4_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
Cx = 6;
N = 5;
d = 2;
%% Time Gap For Inter-Camera Tracking (only #frames)

acc_p_r_EX = [];
acc_p_r_NN = [];
numFramesPi = zeros(length(results_sel),3);
numFramesEX = zeros(length(results_sel),3);
numFramesNN = zeros(length(results_sel),3);
numFramesON = zeros(length(results_sel),3);

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
    
    if isnan(pid.prec )
        disp('');
    end
    acc = [acc; pid.acc];
    P = [P; pid.prec];
    R = [R; pid.recall];
    %C = C + confusionmat(pid.g,pid.p);

    % compute accuracy for exhaustive and NN methods
    gt = pid.g;
    acc_p_r_EX = [acc_p_r_EX; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*N),1.0];
    acc_p_r_NN = [acc_p_r_NN; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*(d+1)),1.0];
    
    % compute time and numFrames for the policy
    numFramesPi(p,:) = compute_num_frames_1_person(pid.p,pid.g, Cx);
    
    % compute time and numFrames for the exhaustive search
    numFramesEX(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'EX');
    
    % compute time and numFrames for the nearest neighbor
    numFramesNN(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'NN');
    
end
% fill number of frames for online-learning approach
numFramesON(:,1) = [10, 25391, 11039, 4406, 20312, 0.0, 4329, 7550, 2506, 0.0, 10908, 0.0, 2681, 0.0, 8, 0.0, 0.0, 13426, 0.0, 0.0, 0.0, 0.0, 25265, 550];
numFramesON(:,2) = [10.0, 36375.0, 16643.0, 10893.0, 24569.0, 0.0, 9754.0, 13432.0, 2355.0, 0.0, 11455.0, 0.0, 25810.0, 0.0, 778.0, 0.0, 0.0, 11567.0, 0.0, 0.0, 0.0, 0.0, 32564.0, 601.0];
numFramesON(:,3) = [1, 3, 2, 1, 3, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0];
numFramesON(:,4) = [0.0, 22134.0, 7608.0, 7524.0, 14004.0, 0.0, 277.0, 3201.0, 1707.0, 0.0, 6973.0, 0.0, 21737.0, 0.0, 777.0, 0.0, 0.0, 5968.0, 0.0, 0.0, 0.0, 0.0, 21974.0, 499.0];


nfPI = numFramesPi;
nfPI(nfPI(:,1)==0,:) = [];
nfEX = numFramesEX;
nfEX(nfEX(:,1)==0,:) = [];
nfNN = numFramesNN;
nfNN(nfNN(:,1)==0,:) = [];
nfON = numFramesON;
nfON(nfON(:,1)==0,:) = [];
% nfON(nfON(:,2)==0,2) = 1;

%nn = numFramesPi;
%nn(nn(:,1)==0,:) = [];
disp( mean([nfEX,nfNN,nfON,nfPI]) );  % *2, because half people were used for testing

% accuracy, precision and recall of other approach for ICT 
disp( mean(nfEX(:,3)./nfEX(:,1)) );
disp( mean(nfEX(:,3)./nfEX(:,2)) );
disp(1);

disp( mean(nfNN(:,3)./nfNN(:,1)) );
disp( mean(nfNN(:,3)./nfNN(:,2)) );
disp(1);

% accuracy, precision and recall of Gaussian approach for ICT 
% % % % disp( mean((nfEX(:,3)+(nfON(:,4)-nfON(:,5)))./nfON(:,1) ));
% % % % disp( mean((nfEX(:,3)-(nfON(:,5)~=0))./nfON(:,2)) );
% % % % disp( mean((nfEX(:,3)-(nfON(:,5)~=0))./nfEX(:,3)) );

% accuracy, precision and recall of Gaussian approach for ICT 
corr = (nfON(:,1)-nfON(:,4));
corr(corr<0) = 0;
disp( mean((corr)./nfON(:,1) ));
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
m = nfEX; c = 1;
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(c),'*'),'MarkerSize',18)
  if c >= 7
      c = 1;
  else
      c = c+1;
  end
end
m = nfNN; c = 1;
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(c),'+'),'MarkerSize',18)
  if c >= 7
      c = 1;
  else
      c = c+1;
  end
end
m = nfON; c = 1;
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(c),'x'),'MarkerSize',18)
  if c >= 7
      c = 1;
  else
      c = c+1;
  end
end
m = nfPI; c = 1;
for i = 1:length(m(:,1))
  plot(m(i,1),m(i,2), strcat(colorstring(c),'o'),'MarkerSize',18)
  if c >= 7
      c = 1;
  else
      c = c+1;
  end
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
disp(C);

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
