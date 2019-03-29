function train_q_db3_aamas_3rew()
%  
% AAMAS-2018, Camera Selection
% Anil Sharma, IIIT-Delhi
%

addpath('train/');
addpath('test/');
addpath(genpath('utils/'));

%addpath(genpath('/media/win/data/Person_reID_baseline_matconvnet/'));

opts.num_camera = 4;
opts.numEpoch = 50000;

task = 1; % 0 for train, 1 for test
%Q_table_name = './models/q_tableRL_db3_pICAPS_3rew_te2_tel200.mat';
Q_table_name = './models/q_tableRL_db3_pICAPS_3rew_te1_tel400.mat';
% opts.num_camera = 5;
if task == 0
    
    load(Q_table_name, 'Qc')  
    [Qc,reward,rew_allep] = train_QcamPolicy_db3_aamas_postICAPS_3rew(opts, Q_table_name,Qc);  
    save(Q_table_name, 'Qc','-v7.3')  
    %save('./results/q_results_db3_pICAPS_3rew_te1_tel400.mat', 'reward','rew_allep');
end
if task == 1
    load(Q_table_name, 'Qc')  
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db3_aamas(opts,Qc);  
    %save('./results_aamas/q_results_db3_aamas_test_exp1_fn30_final.mat', 'reward','rew_allep','results_sel','pred_allP');
    %save('./results_aamas/q_results_db3_aamas_test_exp2_fn15_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_aamas/MTMC_traj_db3_exp1_fn30_aamas', traj);
    csvwrite('./results_aamas/MTMC_traj_db3_exp2_fn15_traj_aamas', traj);

end


disp(''); 


