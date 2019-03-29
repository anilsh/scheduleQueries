function transition_time_distribution_db4

% load dataset file
load('../results_icaps/q_results_db4_icaps_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
Cx = 6;
N = 5;
d = 2;
%% Time Gap For Inter-Camera Tracking (only #frames)
p_4 = [1 46 17 6 11 22 34 26 27 7 8 48 35 25 45 23 47 30 36 41 12 15 21 19  5 39 33 40 42 10 9 13 24 44 29 16 20 49 3 37 31 38 43 28 2 4 14 18 32];
p_4 = p_4(2:2:end);

tau_allp_gt = cell(N,N);
tau_allp_poll = cell(N,N);
for i = 1:length(results_sel)
    pid = p_4(i);
    g = results_sel{i}.g;
    p = pred_allP{pid};
    
    %[taup,taug] = tau_one_seq(p,g, N,Cx);
    tau_allp_gt = tau_gt_fn(g, Cx, tau_allp_gt);
    tau_allp_poll = tau_pr_fn(p, tau_allp_poll);
    
    disp('');
end

gt_mu = zeros(N,N);
p_mu = zeros(N,N);
for ci = 1:N
    for cj = 1:N
        gt_mu(ci,cj) = mean(tau_allp_gt{ci,cj});
        p_mu(ci,cj) = mean(tau_allp_poll{ci,cj});
    end
end

figure;
%histogram(tau_allp_gt{a,b},10,'Normalization','probability', 'FaceColor', 'b');
histogram(tau_allp_gt{a,b},10, 'FaceColor', 'b');
hold on;
%histogram(tau_allp_poll{a,b},10,'Normalization','probability','FaceColor','r');
histogram(tau_allp_poll{a,b},10, 'FaceColor','r');
title('Normalized distribution of distances among positive and negative pairs');
legend('Ground truth','Policy');

disp('');

function tau_all = tau_gt_fn(g, Cx, tau_all)

% add index of all elements to p
g = [g, find(g)];
% remove all entries of Cx
g = g(g(:,1)~=Cx,:);

tau_all = compute_tau(g, tau_all);

function tau_all = tau_pr_fn(p, tau_all)

% create transition vector
%tau_all = cell(N,N);

% take camera sequence and frame#
p = p(:,[1,2]);
tau_all = compute_tau(p, tau_all);

function tau_all = compute_tau(p, tau_all)

for i = 2:size(p,1)-3
    if p(i,1) == p(i-1,1) && p(i,2) == p(i-1,2)+1  % it says same camera
        continue;
    end
    
    % compute transition length
    tau_this_tran = p(i,2)-p(i-1,2);
    if tau_this_tran < 3 %|| any(p(i,1) ~= p(i+1:i+3,1) )
        continue;
    end
    % start and end camera
    start_c = p(i-1,1);
    end_c = p(i,1);
    
    % store the transition value
    tau_all{start_c,end_c} = [tau_all{start_c,end_c}; tau_this_tran];
end
