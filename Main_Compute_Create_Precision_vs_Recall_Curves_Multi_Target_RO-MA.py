import Charts

num_samples = 100

main_path = "./results/results_avg/"


result_path = [
    main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO/',
    main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_RO/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Cerrado_MA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_RO/'
    ]

labels = [
    'Tr: MA,Ts: MA (Source only training)',
    'Tr: RO->MA, Ts: MA (domain adaptation single-target)',
    'Tr: RO->PA,MA Ts: MA (domain adaptation training on multi-target)',
    'Tr: RO->PA,MA(unblcd) Ts: MA (domain adaptation multi-target)',
    'Tr: RO->PA,MA Ts: MA (domain adaptation multi-target)',
    'Tr: RO->PA,MA(unblcd) Ts: RO (domain adaptation multi-target)',
    'Tr: RO->PA,MA Ts: RO (domain adaptation multi-target)',
    'Tr: RO, Ts: MA (Source only training)',
    'Tr: RO->PA,MA Ts: MA (domain adaptation multi-target 2 neurons discriminator)',    
    'Tr: RO->PA,MA Ts: RO (domain adaptation multi-target 2 neurons discriminator)',
    'Tr: RO->PA,MA Ts: MA (domain adaptation multi-target 3 neurons conv discriminator)',    
    'Tr: RO->PA,MA Ts: RO (domain adaptation multi-target 3 neurons conv discriminator)'
    ]

cont = 1
for i in range(0, len(result_path), 5):
    result_path_ = result_path[i : i + 5]
    labels_ = labels[i : i + 5]

    title = 'Multi_Target_Ts_RO_Eval_MA_'+str(cont)
    Charts.create_map_chart(result_path_,labels_,main_path,title,num_samples)
    cont += 1