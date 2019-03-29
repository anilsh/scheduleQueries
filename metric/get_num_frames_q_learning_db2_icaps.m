function get_num_frames_q_learning_db2_icaps

% load dataset file
load('../results_icaps/q_results_db2_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
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

% fill number of frames for online-learning approach
numFramesON(:,1) = [1999.0, 1552.0, 1897.0, 703.0, 1769.0, 0.0, 0.0, 656.0, 836.0, 747.0, 1494.0, 780.0, 790.0, 0.0, 750.0, 787.0, 1509.0, 1888.0, 1592.0, 1569.0, 952.0, 1880.0, 930.0, 738.0, 900.0, 1912.0, 1486.0, 1554.0, 1791.0, 1761.0, 1720.0, 530.0, 2062.0, 1670.0, 1721.0, 644.0, 1843.0, 784.0, 709.0, 762.0, 2742.0, 2347.0, 748.0, 1837.0, 1362.0, 869.0, 1484.0, 637.0, 1718.0, 2047.0, 2014.0, 1854.0, 1584.0, 805.0, 845.0, 1038.0, 747.0, 1107.0, 638.0, 1397.0, 1626.0, 1615.0, 799.0, 807.0, 660.0, 2180.0, 695.0, 715.0, 2365.0, 786.0, 3518.0, 3262.0, 870.0, 1143.0, 826.0, 1865.0, 682.0, 929.0, 704.0, 1236.0, 1704.0, 1226.0, 1869.0, 1695.0, 819.0, 579.0, 1564.0, 2003.0, 1099.0, 959.0, 1952.0, 741.0, 875.0, 907.0, 1450.0, 762.0, 1785.0, 1297.0, 510.0, 871.0, 844.0, 770.0, 1260.0, 1085.0, 1128.0, 1442.0, 1338.0, 980.0, 647.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 922.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 10418.0, 0.0, 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3367.0, 0.0, 2, 132.0, 0.0, 0.0, 0.0, 1, 1, 0.0, 1, 24.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 3712.0, 0.0, 192.0, 0.0, 0.0, 1, 178.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2, 8855.0, 1, 16.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 307.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 39.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 4.0, 0.0, 18.0, 0.0, 9.0, 1, 0.0, 20.0, 1, 167.0, 2, 18.0, 8.0, 16.0, 331.0, 14.0, 26.0, 0.0, 8.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0];
numFramesON(:,2) = [1883.0, 1578.0, 1944.0, 787.0, 1675.0, 0.0, 0.0, 489.0, 785.0, 628.0, 1544.0, 774.0, 744.0, 0.0, 516.0, 642.0, 1437.0, 1572.0, 1398.0, 1634.0, 813.0, 1544.0, 923.0, 707.0, 659.0, 2238.0, 1561.0, 1600.0, 1700.0, 1547.0, 1641.0, 623.0, 2144.0, 1608.0, 1783.0, 694.0, 1662.0, 554.0, 526.0, 696.0, 2453.0, 2514.0, 480.0, 1864.0, 1192.0, 1072.0, 1516.0, 585.0, 2089.0, 1597.0, 1887.0, 2081.0, 1699.0, 1016.0, 876.0, 1147.0, 800.0, 961.0, 625.0, 1560.0, 1543.0, 1664.0, 738.0, 617.0, 623.0, 2257.0, 858.0, 764.0, 2455.0, 787.0, 3253.0, 3535.0, 777.0, 1242.0, 788.0, 1785.0, 821.0, 816.0, 699.0, 1154.0, 2231.0, 1062.0, 1570.0, 2114.0, 880.0, 531.0, 1805.0, 1694.0, 897.0, 1080.0, 2364.0, 734.0, 1001.0, 759.0, 1377.0, 759.0, 1581.0, 1449.0, 421.0, 874.0, 1053.0, 785.0, 1236.0, 1075.0, 1154.0, 1288.0, 1175.0, 1011.0, 574.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 955.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 12628.0, 0.0, 16.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3374.0, 0.0, 7.0, 98.0, 0.0, 0.0, 0.0, 0, 14.0, 0.0, 0, 23.0, 0.0, 7.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 5305.0, 0.0, 192.0, 0.0, 0.0, 47.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 333.0, 9403.0, 0, 15.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 305.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 186.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 4.0, 0.0, 18.0, 0.0, 0, 143.0, 0.0, 0, 18.0, 0, 13.0, 401.0, 0, 0, 305.0, 14.0, 0, 0.0, 8.0, 0, 0, 0.0, 0.0, 0.0, 0.0];
numFramesON(:,3) = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0];
numFramesON(:,4) = [1247.0, 1669.0, 1765.0, 924.0, 1579.0, 0.0, 0.0, 734.0, 1092.0, 820.0, 1710.0, 746.0, 724.0, 0.0, 856.0, 809.0, 1677.0, 2190.0, 1684.0, 1429.0, 412.0, 1651.0, 936.0, 742.0, 606.0, 1410.0, 1599.0, 1358.0, 1223.0, 1588.0, 1155.0, 564.0, 1501.0, 1621.0, 1515.0, 664.0, 1512.0, 651.0, 738.0, 792.0, 1490.0, 1477.0, 991.0, 1741.0, 727.0, 973.0, 1753.0, 1065.0, 1852.0, 1689.0, 1506.0, 1953.0, 804.0, 714.0, 870.0, 886.0, 795.0, 1748.0, 733.0, 1522.0, 1752.0, 1748.0, 834.0, 575.0, 705.0, 1634.0, 752.0, 852.0, 1580.0, 626.0, 1521.0, 1757.0, 606.0, 819.0, 870.0, 1416.0, 634.0, 810.0, 815.0, 640.0, 1394.0, 704.0, 1467.0, 1484.0, 549.0, 599.0, 1592.0, 1637.0, 723.0, 849.0, 1828.0, 723.0, 657.0, 623.0, 1321.0, 798.0, 1627.0, 1820.0, 720.0, 741.0, 768.0, 753.0, 686.0, 873.0, 821.0, 675.0, 754.0, 765.0, 710.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 814.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

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
    %confm = confusionmat(pid.g,pid.p);
    %C = C + confm;
    %C = confusion(pid.g,pid.p);

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
nfON(nfON(:,2)==0,2) = 1;

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
% disp( mean(nfEX(:,3)./(nfON(:,1)-(nfON(:,2)/(d+1)))) );
% disp( mean(nfEX(:,3)./nfON(:,2)) );
% disp(1);

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

