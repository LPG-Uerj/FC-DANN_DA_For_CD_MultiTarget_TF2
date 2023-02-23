import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"

result_path = [
    main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/',
    main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Cerrado_MA/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA/'
]

labels = [
    'Tr: MA,Ts: MA (Source only training)',
    'Tr: PA->MA, Ts: MA (domain adaptation single-target)',
    'Tr: PA->RO,MA Ts: MA (domain adaptation training on multi-target)',
    'Tr: PA->RO,MA(unblcd) Ts: MA (domain adaptation multi-target)',
    'Tr: PA->RO,MA Ts: MA (domain adaptation multi-target)',
    'Tr: PA->RO,MA(unblcd) Ts: PA (domain adaptation multi-target)',
    'Tr: PA->RO,MA Ts: PA (domain adaptation multi-target)',
    'Tr: PA,Ts: MA (Source only training)',
    'Tr: PA->RO,MA Ts: MA (domain adaptation multi-target 2 neurons discriminator)',    
    'Tr: PA->RO,MA Ts: PA (domain adaptation multi-target 2 neurons discriminator)'
]

for i in range(0, len(result_path), 5):
        result_path_ = result_path[i : i + 5]
        labels_ = labels[i : i + 5]

        title = 'Multi_Target_Ts_PA_Eval_MA_'+str(i)
        Charts.create_map_chart(result_path_,labels_,main_path,title,num_samples)