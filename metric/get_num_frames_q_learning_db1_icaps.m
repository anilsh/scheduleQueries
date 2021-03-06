function get_num_frames_q_learning_db1_icaps

% load dataset file
load('../results_icaps/q_results_db1_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
Cx = 4;
N = 3;
d = 1;
%% Time Gap For Inter-Camera Tracking (only #frames)

acc_p_r_EX_both = [];
acc_p_r_NN_both = [];

acc_p_r_ON = [];
%nnfPI = zeros(length(results_sel),2);
numFramesPi = zeros(length(results_sel),3);
numFramesEX = zeros(length(results_sel),3);
numFramesNN = zeros(length(results_sel),3);
numFramesON = zeros(length(results_sel),3);

% fill number of frames for online-learning approach (results from the
% python script)
numFramesON(:,1) = [1, 0.0, 967.0, 2004.0, 0.0, 0.0, 1783.0, 968.0, 0.0, 3, 1418.0, 0.0, 0.0, 880.0, 1213.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1516.0, 1543.0, 749.0, 1672.0, 780.0, 0.0, 1, 0.0, 8.0, 0.0, 0.0, 798.0, 8.0, 0.0, 0.0, 1, 9.0, 2092.0, 898.0, 0.0, 0.0, 7.0, 159.0, 718.0, 1, 0.0, 1, 342.0, 1, 0.0, 1044.0, 0.0, 0.0, 0.0, 1, 0.0, 1078.0, 1, 0.0, 653.0, 0.0, 0.0, 0.0, 1, 0.0, 9464.0, 629.0, 0.0, 0.0, 853.0, 0.0, 0.0, 1135.0, 860.0, 0.0, 0.0, 0.0, 2, 801.0, 710.0, 0.0, 1860.0, 0.0, 0.0, 0.0, 2033.0, 1598.0, 0.0, 0.0, 772.0, 0.0, 0.0, 0.0, 898.0, 2100.0, 0.0, 1029.0, 1165.0, 0.0, 0.0, 1179.0, 0.0, 0.0, 6.0, 659.0, 0.0, 1171.0, 0.0, 0.0, 0.0, 805.0, 7854.0];
numFramesON(:,2) = [66.0, 0.0, 1176.0, 2114.0, 0.0, 0.0, 1997.0, 1099.0, 0.0, 12.0, 1131.0, 0.0, 0.0, 943.0, 980.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 211.0, 1511.0, 1648.0, 750.0, 1981.0, 841.0, 0.0, 1, 0.0, 13.0, 0.0, 0.0, 790.0, 8.0, 0.0, 0.0, 1, 1, 2046.0, 856.0, 0.0, 0.0, 2, 159.0, 634.0, 1, 0.0, 1, 342.0, 1, 0.0, 1235.0, 0.0, 0.0, 0.0, 1, 0.0, 1087.0, 1, 0.0, 643.0, 0.0, 0.0, 0.0, 1, 0.0, 12121.0, 625.0, 0.0, 0.0, 840.0, 0.0, 0.0, 975.0, 783.0, 0.0, 0.0, 0.0, 18.0, 800.0, 712.0, 0.0, 1648.0, 0.0, 0.0, 0.0, 2101.0, 1587.0, 0.0, 0.0, 764.0, 0.0, 0.0, 0.0, 1006.0, 1796.0, 0.0, 1081.0, 1120.0, 0.0, 0.0, 1318.0, 0.0, 0.0, 1, 659.0, 0.0, 1102.0, 0.0, 0.0, 0.0, 787.0, 7644.0];
numFramesON(:,3) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1];
numFramesON(:,4) = [0.0, 0.0, 1055.0, 1949.0, 0.0, 0.0, 1997.0, 768.0, 0.0, 0.0, 848.0, 0.0, 0.0, 908.0, 617.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1706.0, 1995.0, 787.0, 1850.0, 782.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 953.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1675.0, 794.0, 0.0, 0.0, 0.0, 0.0, 558.0, 0.0, 0.0, 0.0, 825.0, 0.0, 0.0, 856.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1014.0, 0.0, 0.0, 661.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 929.0, 0.0, 0.0, 836.0, 0.0, 0.0, 783.0, 704.0, 0.0, 0.0, 0.0, 0.0, 871.0, 1000.0, 0.0, 1615.0, 0.0, 0.0, 0.0, 1762.0, 798.0, 0.0, 0.0, 762.0, 0.0, 0.0, 0.0, 890.0, 1679.0, 0.0, 774.0, 815.0, 0.0, 0.0, 871.0, 0.0, 0.0, 0.0, 819.0, 0.0, 685.0, 0.0, 0.0, 0.0, 755.0, 0.0];


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
    
    pr = pid.p;
    gt = pid.g;
    
    % compute accuracy for exhaustive and NN methods
    acc_p_r_EX_both = [acc_p_r_EX_both; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*N),1.0];
    acc_p_r_NN_both = [acc_p_r_NN_both; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*(d+1)),1.0];
    %acc_p_r_EX_ict = [acc_p_r_EX_ict; nnz(gt~=Cx)/length(gt==Cx),nnz(gt~=Cx)/(length(gt==Cx)*N),1.0];
    %acc_p_r_NN_ict = [acc_p_r_NN_ict; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx)/(length(gt)*(d+1)),1.0];
    
    acc_p_r_ON = [acc_p_r_ON; nnz(gt~=Cx)/length(gt),nnz(gt~=Cx),1.0];
    
    % compute time and numFrames for the policy
    nnfPi(p,:) = sum(gt==Cx & pr~=Cx) + sum(pr~=gt & gt~=Cx & pr~=Cx);
    numFramesPi(p,:) = compute_num_frames_1_person(pid.p,pid.g, Cx);
    
    % compute time and numFrames for the exhaustive search
    numFramesEX(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'EX');
    
    % compute time and numFrames for the nearest neighbor
    numFramesNN(p,:) = compute_num_frames_ex_nn(pid.g, Cx,N,d, 'NN');
    
end


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
disp( mean([nfPI,nfEX,nfNN]) );  % *2, because half people were used for testing

% accuracy, precision and recall of other approach for ICT 
disp( mean(nfEX(:,3)./nfEX(:,1)) );
disp( mean(nfEX(:,3)./nfEX(:,2)) );
disp(1);

disp( mean(nfNN(:,3)./nfNN(:,1)) );
disp( mean(nfNN(:,3)./nfNN(:,2)) );
disp(1);

% % accuracy, precision and recall of Gaussian approach for ICT 
% disp( mean((nfEX(:,3)+(nfON(:,4)-nfON(:,5)))./nfON(:,1) ));
% disp( mean((nfEX(:,3)-(nfON(:,5)~=0))./nfON(:,2)) );
% disp( mean((nfEX(:,3)-(nfON(:,5)~=0))./nfEX(:,3)) );

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

disp();

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

