function APR_EX_NN
% Compute A, P, R for exhaustive and neighbor approach
% Output format: 4x4
%     1st column: Exhaustive (SCT+ICT)
%     2nd column: Exhaustive (GT+ICT), i.e., GT is used for SCT
%     3rd column: Neighbor (SCT+ICT)
%     4th column: Neighbor (GT+ICT), i.e., GT is used for SCT
%     1st row: Accuracy (A)
%     2nd row: Precision (P)
%     3rd row: Recall (R)
%     4th row: averageNumberFrames (F)
%

db = 4;
if db == 3
    load('../results_icaps/q_results_db3_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    Cx = 5;
    N = 4;
    d = 2;

elseif db == 4
    load('../results_icaps/q_results_db4_aamas_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    Cx = 6;
    N = 5;
    d = 2;
end


A = zeros(1,4);  % EX_both, EX_ict, NN_both, NN_ICT
P = zeros(1,4);
R = zeros(1,4);
F = zeros(1,4);
% For all person
for p = 1:length(results_sel)
    
    pid = results_sel{p};
    if isempty(pid)
        continue;
    end
    g = single(pid.g);
    
    [a1,p1,r1,f1] = compute_APR_EX_SCT_ICT(g, N,Cx);
    [a2,p2,r2,f2] = compute_APR_EX_gtSCT(g, N,Cx);
    
    [a3,p3,r3,f3] = compute_APR_NN_SCT_ICT(g, d,Cx);
    [a4,p4,r4,f4] = compute_APR_NN_gtSCT(g, d,Cx);
    
    A = A + [a1,a2,a3,a4];
    P = P + [p1,p2,p3,p4];
    R = R + [r1,r2,r3,r4];
    F = F + [f1,f2,f3,f4];
    
end
disp(A/length(results_sel));
disp(P/length(results_sel));
disp(R/length(results_sel));
disp(F/length(results_sel));
disp('');


function [A,P,R,F] = compute_APR_EX_SCT_ICT(g, N,Cx)

A = sum(g~=Cx) / length(g);
P = sum(g~=Cx) / (N*length(g));
R = sum(g~=Cx) / sum(g ~= Cx);

F = N*sum(g==Cx) + (N-1)*sum(g~=Cx);

function [A,P,R,F] = compute_APR_EX_gtSCT(g, N,Cx)

A = sum(g~=Cx) / length(g);
P = sum(g~=Cx) / (N*sum(g==Cx) + sum(g~=Cx));
R = sum(g~=Cx) / sum(g ~= Cx);

F = N*sum(g==Cx); % + N #transitions


function [A,P,R,F] = compute_APR_NN_SCT_ICT(g, d,Cx)

A = sum(g~=Cx) / length(g);
P = sum(g~=Cx) / ((d+1)*length(g));
R = sum(g~=Cx) / sum(g ~= Cx);

F = (d+1)*sum(g==Cx) + (d+1-1)*sum(g~=Cx);

function [A,P,R,F] = compute_APR_NN_gtSCT(g, d,Cx)

A = sum(g~=Cx) / length(g);
P = sum(g~=Cx) / ((d+1)*sum(g==Cx) + sum(g~=Cx));
R = sum(g~=Cx) / sum(g ~= Cx);

F = (d+1)*sum(g==Cx); % + N #transitions

