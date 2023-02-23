import Charts

num_samples = 100

main_path = "./results/results_avg/"

result_path = [
    main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DRCL_multi_balanced_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Cerrado_MA/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Cerrado_MA/',
    main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_RO/',
    main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_domain_labels_False_Cerrado_MA/'
]


labels = [
    'Tr: RO,Ts: RO (Source only training)',
    'Tr: MA->RO, Ts: RO (domain adaptation single-target)',
    'Tr: MA->RO,PA, Ts: RO (domain adaptation training on multi-target)',
    'Tr: MA->RO,PA(unblcd) Ts: RO (domain adaptation multi-target)',
    'Tr: MA->RO,PA Ts: RO (domain adaptation multi-target)',
    'Tr: MA->RO,PA(unblcd) Ts: MA (domain adaptation multi-target)',
    'Tr: MA->RO,PA Ts: MA (domain adaptation multi-target)',
    'Tr: MA,Ts: RO (Source only training)',
    'Tr: MA->RO,PA Ts: RO (domain adaptation multi-target 2 neurons discriminator)',    
    'Tr: MA->RO,PA Ts: MA (domain adaptation multi-target 2 neurons discriminator)',
    ]


for i in range(0, len(result_path), 5):
        result_path_ = result_path[i : i + 5]
        labels_ = labels[i : i + 5]

        title = 'Multi_Target_Ts_MA_Eval_RO_'+str(i)
        Charts.create_map_chart(result_path_,labels_,main_path,title,num_samples)