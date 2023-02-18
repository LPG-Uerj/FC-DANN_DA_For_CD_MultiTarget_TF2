import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = RO, Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/')

#X = MA->RO, Y = RO
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/')

#Multi
#X = MA->RO,PA, Y = RO
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DRCL_multi_balanced_Amazon_RO/')

#X = MA->RO,PA, Y = RO
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_RO/')

#X = MA->RO,PA, Y = RO
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Amazon_RO/')

#X = MA->RO,PA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Cerrado_MA/')

#X = MA->RO,PA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Cerrado_MA/')

#X = MA, Y = RO
result_path.append(main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/')


labels = []
labels.append('Tr: RO,Ts: RO (Source only training)')
labels.append('Tr: MA->RO, Ts: RO (domain adaptation single-target)')
labels.append('Tr: MA->RO,PA, Ts: RO (domain adaptation training on multi-target)')
labels.append('Tr: MA->RO,PA, Ts: RO (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA(blcd) Ts: RO (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA Ts: MA (domain adaptation multi-target)')
labels.append('Tr: MA->RO,PA(blcd) Ts: MA (domain adaptation multi-target)')
labels.append('Tr: MA,Ts: RO (Source only training)')


title = 'Multi_Target_Ts_MA_Eval_RO'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)