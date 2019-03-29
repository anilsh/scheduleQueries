function train_q_db4_icaps()
%  
% ICAPS-2019, Camera Selection
% Anil Sharma, IIIT-Delhi
%

addpath('train/');
addpath(genpath('utils/'));

addpath(genpath('/media/win/data/Person_reID_baseline_matconvnet/'));

opts.num_camera = 5;
opts.numEpoch = 1000;
opts.fpath = '/media/win/data/MCT dataset/dataset4';

% NOTE: with _100, exploration changed

task = 3; % 0 for train, 1 for test
% Q_table_name = './models/q_tableRL_db4_icaps_100.mat';
Q_table_name = './models/q_tableRL_db4_icaps.mat';
if task == 0
    
    load(Q_table_name , 'Qc')  
    if exist('Qc')
        [Qc,reward,rew_allep] = train_QcamPolicy_db4_icaps(opts, Q_table_name, Qc);
    else
        [Qc,reward,rew_allep] = train_QcamPolicy_db4_icaps(opts, Q_table_name);
    end
    save(Q_table_name , 'Qc','-v7.3')  
    save('./results/q_results_db4_icaps_100.mat', 'reward','rew_allep');

end

% for testing with ground-truth as presence block
if task == 1
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel,pred_allP] = test_QcamPolicy_db4_icaps(opts,Qc);  
    %save('./results_icaps/q_results_db4_icaps_test_exp1_fn20_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    save('./results_icaps/q_results_db4_icaps_test_exp2_fn30_seq_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    
    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_icaps/MTMC_traj_db4_exp1_fn20_icaps', traj);
    csvwrite('./results_icaps/MTMC_traj_db4_exp2_fn30_seq_icaps', traj);
end

% for testing with errors in re-identification (included in paper)
if task == 3
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel,pred_allP] = test_QcamPolicy_db4_bboxSeq_icaps(opts,Qc);  
    save('./results_icaps/q_results_db4_icaps_test_exp1_err20Seq_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    
    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_icaps/MTMC_traj_db4_exp1_fn20_icaps', traj);
    csvwrite('./results_icaps/MTMC_traj_db4_exp1_err20Seq_traj_icaps', traj);
end


%[Qc,MCTA,results_det] = train_RL_QcamPolicy_longTE_final_db3_exp1(opts, Qc);  
%save('./results/q_tableRL_final_db3_d8_exp1_test.mat', 'results_det','MCTA');

disp(''); 


