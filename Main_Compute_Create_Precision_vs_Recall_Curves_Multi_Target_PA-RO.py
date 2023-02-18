import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = RO, Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/')

#X = PA->RO, Y = RO
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/')

#X = PA->RO,MA Y = RO
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_RO/')

#X = PA->RO,MA Y = RO
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO/')

#X = PA->RO,MA(blcd) Y = RO
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO/')

#X = PA->RO,MA Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/')

#X = PA->RO,MA(blcd) Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/')

#X = PA, Y = RO
result_path.append(main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_RO/')


labels = []
labels.append('Tr: RO,Ts: RO (Source only training)')
labels.append('Tr: PA->RO, Ts: RO (domain adaptation single-target)')
labels.append('Tr: PA->RO,MA Ts: RO (domain adaptation training on multi-target)')
labels.append('Tr: PA->RO,MA Ts: RO (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA(blcd) Ts: RO (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA Ts: PA (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA(blcd) Ts: PA (domain adaptation multi-target)')
labels.append('Tr: PA,Ts: RO (Source only training)')

title = 'Multi_Target_Ts_PA_Eval_RO'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)