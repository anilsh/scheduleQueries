function train_q_db1_icaps()
%  
% ICAPS 2019, Camera Selection
% Anil Sharma, IIIT-Delhi
%
% Please edit variable 'task' for training/testing/erroneous testing. 
task = 1; % 0 for train, 1 for test, more for test with errors in re-id

addpath('train/');
addpath('test/');
addpath(genpath('utils/'));

opts.num_camera = 3;
opts.numEpoch = 1;
opts.fpath = '/media/win/data/MCT dataset/dataset1';

Q_table_name = './models/q_tableRL_db1_icaps_2.mat';
% task=0, to resume training or to train the model from scratch
if task == 0
    %load('./models/q_tableRL_db1_icaps_2.mat', 'Qc'); % comment this line if want to train from scratch
    if exist('Qc')
        [Qc,reward,rew_allep] = train_QcamPolicy_db1_icaps(opts,Q_table_name,Qc);  
    else
        [Qc,reward,rew_allep] = train_QcamPolicy_db1_icaps(opts,Q_table_name);  
    end
    save(Q_table_name, 'Qc','-v7.3')  
    save('./results_icaps/q_results_db1_icaps_2.mat', 'reward','rew_allep');
end

% task=1, to test the trained model on test-set with GT for
% re-identification
if task == 1
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db1_icaps(opts,Qc);  
    save('./results_icaps/q_results_db1_icaps_test_exp2_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    % combine trajectory for all targets for MCT evaluation
    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    csvwrite('./results_icaps/MTMC_traj_db1_exp2_gt_traj_icaps', traj);

end

% task=3, to test the trained model on test-set with errors in 
% re-identification
if task == 3
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db1_bboxSeq_icaps(opts,Qc);  
    save('./results_icaps/q_results_db1_icaps_test_exp1_err20Seq_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    % combine trajectory for all targets for MCT evaluation
    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    csvwrite('./results_icaps/MTMC_traj_db1_exp1_err20Seq_traj_icaps', traj);

end

% task=3, to test the trained model on db2 with errors in 
% re-identification. Db-1 and Db-2 has same network topology and hence no
% train-test split is created for db-2. 
if task == 4  % only for dataset-2
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db2_bboxSeq_icaps(opts,Qc);  
    save('./results_icaps/q_results_db2_icaps_test_exp1_gt_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    % combine trajectory for all targets for MCT evaluation
    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    csvwrite('./results_icaps/MTMC_traj_db2_exp1_gt_traj_icaps', traj);

end

%[Qc,MCTA,results_det] = train_RL_QcamPolicy_longTE_final_db3_exp1(opts, Qc);  
%save('./results/q_tableRL_final_db3_d8_exp1_test.mat', 'results_det','MCTA');

disp(''); 


