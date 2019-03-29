function train_q_db1_aamas_selectNearTr_te2_f2()
%  
% AAMAS-2018, Camera Selection
% Anil Sharma, IIIT-Delhi
%

addpath('train/');
addpath('test/');
addpath(genpath('utils/'));

%addpath(genpath('/media/win/data/Person_reID_baseline_matconvnet/'));

opts.num_camera = 3;
opts.numEpoch = 50000;

task = 0; % 0 for train, 1 for test
% Q_table_name = './models/q_tableRL_db3_aamas_200.mat';
Q_table_name = './models/q_tableRL_db1_pICAPS_te2_tel400_selectNearTr_el100_largeR_fpsc2.mat';
% opts.num_camera = 5;
if task == 0
    
    %load('./models/.mat', 'Qc')  
    if exist('Qc')
        [Qc,reward,rew_allep] = train_db1_postICAPS_3rew_selectNearTr_te2_f2(opts, Q_table_name,Qc);  
    else
        [Qc,reward,rew_allep] = train_db1_postICAPS_3rew_selectNearTr_te2_f2(opts, Q_table_name);  
    end
    save(Q_table_name, 'Qc','-v7.3')  
    
end
if task == 1
    load(Q_table_name, 'Qc')  
    
    [reward,rew_allep,results_sel, pred_allP] = test_QcamPolicy_db1_aamas(opts,Qc);  
    %save('./results_aamas/q_results_db3_aamas_test_exp2_fn15_traj_final.mat', 'reward','rew_allep','results_sel','pred_allP');

    traj = [];
    for i = 1: length(pred_allP)
        traj = [traj; pred_allP{i}];
    end
    %csvwrite('./results_aamas/MTMC_traj_db1_exp1_fn30_aamas', traj);
    %csvwrite('./results_aamas/MTMC_traj_db1_exp2_fn15_traj_aamas', traj);

end


disp(''); 


