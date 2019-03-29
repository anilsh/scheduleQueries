function train_q_db3_icaps()
%  
% ICAPS-2019, Camera Selection
% Anil Sharma, IIIT-Delhi
%

task = 1; % 0 for train, 1 for test

addpath('train/');
addpath('test/');
addpath(genpath('utils/'));

%addpath(genpath('/media/win/data/Person_reID_baseline_matconvnet/'));

opts.num_camera = 4;
opts.numEpoch = 10000;
opts.fpath = '/media/win/data/MCT dataset/dataset3';


Q_table_name = './models/q_tableRL_db3_icaps_200.mat';
%Q_table_name = './models/q_tableRL_db3_pICAPS_te1_tel200_2.mat';
if task == 0
    
    %load('./models/q_tableRL_db3_pICAPS_te1_tel200.mat', 'Qc')  
    if exist('Qc')
        [Qc,reward,rew_allep] = train_QcamPolicy_db3_icaps(opts, Q_table_name, Qc);  
    else
        [Qc,reward,rew_allep] = train_QcamPolicy_db3_icaps(opts, Q_table_name);  
    end
    %[Qc,reward,rew_allep] = train_QcamPolicy_db3_icaps_postICAPS(opts, Q_table_name,Qc);  
    save(Q_table_name, 'Qc','-v7.3')  
    save('./results_icaps/q_results_db3_pICAPS_te1_tel200_2.mat', 'reward','rew_allep');
end

% for testing
if task == 1
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db3_icaps(opts,Qc);  
    %save('./results_icaps/q_results_db3_icaps_test_exp1_fn30_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    %save('./results_icaps/q_results_db3_icaps_test_exp2_fn15_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    save('./results_icaps/q_results_db3_icaps_train_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_icaps/MTMC_traj_db3_exp1_fn30_icaps', traj);
    csvwrite('./results_icaps/MTMC_traj_db3_exp2_fn15_traj_icaps', traj);

end

% for testing with errors in re-identification (False rejection, not
% included in paper)
if task == 2
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db3_errREIDicaps(opts,Qc);  
    %save('./results_icaps/q_results_db3_icaps_test_exp1_fn30_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    save('./results_icaps/q_results_db3_icaps_test_exp2_err5_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_icaps/MTMC_traj_db3_exp1_fn30_icaps', traj);
    csvwrite('./results_icaps/MTMC_traj_db3_exp2_err5_traj_icaps', traj);

end

% for testing with errors in re-identification (included in paper)
if task == 3
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db3_bboxSeq_icaps(opts,Qc);  
    save('./results_icaps/q_results_db3_icaps_test_exp1_err20Seq_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    csvwrite('./results_icaps/MTMC_traj_db3_exp1_err20Seq_traj_icaps', traj);

end


disp(''); 


