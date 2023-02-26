import Charts

num_samples = 100

main_path = "./results/results_avg/"

result_path= [
    main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/',
    main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA/']

labels = [
    'Tr: RO,Ts: RO (Source only training)',
    'Tr: PA->RO, Ts: RO (domain adaptation single-target)',
    'Tr: PA->RO,MA Ts: RO (domain adaptation training on multi-target)',
    'Tr: PA->RO,MA(unblcd) Ts: RO (domain adaptation multi-target)',
    'Tr: PA->RO,MA Ts: RO (domain adaptation multi-target)',
    'Tr: PA->RO,MA(unblcd) Ts: PA (domain adaptation multi-target)',
    'Tr: PA->RO,MA Ts: PA (domain adaptation multi-target)',
    'Tr: PA,Ts: RO (Source only training)',
    'Tr: PA->RO,MA Ts: RO (domain adaptation multi-target 2 neurons discriminator)',    
    'Tr: PA->RO,MA Ts: PA (domain adaptation multi-target 2 neurons discriminator)',
    'Tr: PA->RO,MA Ts: RO (domain adaptation multi-target 3 neurons conv discriminator)',    
    'Tr: PA->RO,MA Ts: PA (domain adaptation multi-target 3 neurons conv discriminator)'
]

cont = 1
for i in range(0, len(result_path), 5):
    result_path_ = result_path[i : i + 5]
    labels_ = labels[i : i + 5]

    title = 'Multi_Target_Ts_PA_Eval_RO_'+str(cont)
    Charts.create_map_chart(result_path_,labels_,main_path,title,num_samples)
    cont += 1