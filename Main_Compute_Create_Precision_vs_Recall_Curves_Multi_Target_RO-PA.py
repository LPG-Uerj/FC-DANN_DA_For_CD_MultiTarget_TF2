import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = PA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/')

#X = RO->PA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/')

#Multi
#X = RO->PA,MA Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA/')

#X = RO->PA,MA Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/')

#X = RO->PA,MA(blcd) Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/')

#X = RO->PA,MA Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO/')

#X = RO->PA,MA(blcd) Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO/')

#X = RO, Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/')

labels = []
labels.append('Tr: PA,Ts: PA (Source only training)')
labels.append('Tr: RO->PA, Ts: PA (domain adaptation single-target)')
labels.append('Tr: RO->PA,MA Ts: PA (domain adaptation training on multi-target)')
labels.append('Tr: RO->PA,MA Ts: PA (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA(blcd) Ts: PA (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA Ts: RO (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA(blcd) Ts: RO (domain adaptation multi-target)')
labels.append('Tr: RO,Ts: PA (Source only training)')

title = 'Multi_Target_Ts_RO_Eval_PA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)