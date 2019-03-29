function transition_GT_db4

% load dataset file
%load('./ann_MCT_dataset4_pidWise.mat');
%N = 5;

load('./ann_MCT_dataset3_pidWise.mat');
N = 4;

%load('/media/win/data/dukeMTMC/ann_DukeDataset_pidWise.mat');
%N = 8;

%% Time Gap For Inter-Camera Tracking (only #frames)
%pALL = [1 46 17 6 11 22 34 26 27 7 8 48 35 25 45 23 47 30 36 41 12 15 21 19  5 39 33 40 42 10 9 13 24 44 29 16 20 49 3 37 31 38 43 28 2 4 14 18 32];
%p_3 = pALL(2:2:end);

pALL = randperm(length(PID));

tau_allp_tr = cell(N,N);
tau_allp_test = cell(N,N);
for i = 1:length(PID)
    pid = pALL(i);
    g = PID{pid}(:,[1,2]);
    
    if mod(i,2) ~= 0 % training
        tau_allp_tr = compute_tau(g, tau_allp_tr);
    else   % testing
        tau_allp_test = compute_tau(g, tau_allp_test);
    end
    
    disp('');
end

tr_mu = zeros(N,N);
test_mu = zeros(N,N);
for ci = 1:N
    for cj = 1:N
        tr_mu(ci,cj) = mean(tau_allp_tr{ci,cj});
        test_mu(ci,cj) = mean(tau_allp_test{ci,cj});
    end
end

figure;
a = 1; b = 2;
%histogram(tau_allp_gt{a,b},10,'Normalization','probability', 'FaceColor', 'b');
histogram(tau_allp_tr{a,b},10, 'FaceColor', 'b');
hold on;
%histogram(tau_allp_poll{a,b},10,'Normalization','probability','FaceColor','r');
histogram(tau_allp_test{a,b},10, 'FaceColor','r');
title('Normalized distribution of distances among positive and negative pairs');
legend('Training','Testing');

disp('');


function tau_all = compute_tau(p, tau_all)

for i = 2:size(p,1)-3
    if p(i,1) == p(i-1,1) && p(i,2) == p(i-1,2)+1  % it says same camera
        continue;
    end
    if p(i,1) ~= p(i-1,1) && p(i,2) == p(i-1,2)  % it says same camera
        continue;
    end
    
    % compute transition length
    tau_this_tran = p(i,2)-p(i-1,2);
    % start and end camera
    start_c = p(i-1,1);
    end_c = p(i,1);
    
    % store the transition value
    if tau_this_tran <= 2
        continue;
        disp('');
    end
    tau_all{start_c,end_c} = [tau_all{start_c,end_c}; tau_this_tran];
end
