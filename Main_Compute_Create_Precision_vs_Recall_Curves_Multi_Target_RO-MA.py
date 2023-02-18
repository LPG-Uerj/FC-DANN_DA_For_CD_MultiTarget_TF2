import Charts

num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = MA, Y = MA
result_path.append(main_path + 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/')
#X = RO->MA, Y = MA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/')

#Multi
#X = RO->PA,MA Y = MA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/')

#X = RO->PA,MA Y = MA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/')

#X = RO->PA,MA(blcd) Y = MA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/')

#Multi
#X = RO->PA,MA Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO/')

#X = RO->PA,MA(blcd) Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO/')

#X = RO, Y = MA
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/')


labels = []
labels.append('Tr: MA,Ts: MA (Source only training)')
labels.append('Tr: RO->MA, Ts: MA (domain adaptation single-target)')
labels.append('Tr: RO->PA,MA Ts: MA (domain adaptation training on multi-target)')
labels.append('Tr: RO->PA,MA Ts: MA (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA(blcd) Ts: MA (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA Ts: RO (domain adaptation multi-target)')
labels.append('Tr: RO->PA,MA(blcd) Ts: RO (domain adaptation multi-target)')
labels.append('Tr: RO, Ts: MA (Source only training)')

title = 'Multi_Target_Ts_RO_Eval_MA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)