import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = PA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/')

#X = MA->PA, Y = PA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/')

#Multi
#X = MA->RO,PA, Y = PA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DRCL_multi_balanced_Amazon_PA/')

#X = MA->RO,PA, Y = PA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_PA/')

#X = MA->RO,PA(blcd) Y = PA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Amazon_PA/')

#X = MA->RO,PA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Cerrado_MA/')

#X = MA->RO,PA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Cerrado_MA/')

#X = MA, Y = PA
result_path.append(main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/')


labels = []
labels.append('Tr: PA,Ts: PA (Source only training)')
labels.append('Tr: MA->PA, Ts: PA (domain adaptation single-target)')
labels.append('Tr: MA->RO,PA, Ts: PA (domain adaptation training on multi-target)')
labels.append('Tr: MA->RO,PA, Ts: PA (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA(blcd) Ts: PA (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA Ts: MA (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA(blcd) Ts: MA (domain adaptation multi-target)')
labels.append('Tr: MA,Ts: PA (Source only training)')

title = 'Multi_Target_Ts_MA_Eval_PA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)