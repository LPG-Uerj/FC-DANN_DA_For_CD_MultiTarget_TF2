import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = MA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/')
#X = PA->MA, Y = MA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/')

#Multi
#X = PA->RO,MA, Y = MA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/')

#X = PA->RO,MA, Y = MA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/')

#X = PA->RO,MA(blcd) Y = MA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/')

#X = PA->RO,MA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/')

#X = PA->RO,MA(blcd) Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/')

#X = PA, Y = MA
result_path.append(main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/')


labels = []
labels.append('Tr: MA,Ts: MA (Source only training)')
labels.append('Tr: PA->MA, Ts: MA (domain adaptation single-target)')
labels.append('Tr: PA->RO,MA Ts: MA (domain adaptation training on multi-target)')
labels.append('Tr: PA->RO,MA Ts: MA (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA(blcd) Ts: MA (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA Ts: PA (domain adaptation multi-target)')
labels.append('Tr: PA->RO,MA(blcd) Ts: PA (domain adaptation multi-target)')
labels.append('Tr: PA,Ts: MA (Source only training)')

title = 'Multi_Target_Ts_PA_Eval_MA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)