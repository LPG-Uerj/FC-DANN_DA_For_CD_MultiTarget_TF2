import Charts
import argparse
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

parser = argparse.ArgumentParser([])

args = parser.parse_args()

num_samples = 100

main_path = "./results/results_avg/"

lower_bound_path = 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/'
lower_bound_label = 'Tr:RO Ts:MA\n(Source only training)'
lower_bound_checkpoint = 'checkpoint_tr_Amazon_RO_classification_Amazon_RO/'

upper_bound_source_only_path = 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/'
upper_bound_source_only_label = 'Tr:MA Ts:MA\n(Source only training)'
upper_bound_source_only_checkpoint = 'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/'

upper_bound_da_path = 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/'
upper_bound_da_label = 'Tr:RO->PA,MA Ts:MA\n(DA training on multi-target)'
upper_bound_da_checkpoint = 'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA_Cerrado_MA/'

single_target_path = 'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/'
single_target_label = 'Tr:RO->MA Ts:MA\n(DA single-target)'
single_target_checkpoint = 'checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/'


result_path = [   
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/',    
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Cerrado_MA/',    
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_Cerrado_MA/'    
]

labels = [    
    'Tr:RO->PA,MA(unblcd) Ts:MA\n(DA multi-target 3 neurons fc discr.)',
    'Tr:RO->PA,MA Ts:MA\n(DA multi-target 3 neurons fc discr.)',    
    'Tr:RO->PA,MA Ts:MA\n(DA multi-target 2 neurons discr.)',        
    'Tr:RO->PA,MA Ts:MA\n(DA multi-target 3 neurons conv discr.)',
    'Tr:RO->PA,MA Ts:MA\n(DA multi-target 3 neurons fc discr.\n2 runs warmup)'
]

checkpoint_list = [    
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA_Cerrado_MA/',    
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_Amazon_PA_Cerrado_MA/'
]

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = "./results/results/"

target = CERRADO_MA.DATASET

cont = 1
for i in range(0, len(result_path)):
    result_path_ = [upper_bound_source_only_path,upper_bound_da_path,lower_bound_path,single_target_path,result_path[i]]
    labels_ = [upper_bound_source_only_label,upper_bound_da_label,lower_bound_label,single_target_label,labels[i]]
    checkpoint_list_ = [upper_bound_source_only_checkpoint,upper_bound_da_checkpoint,lower_bound_checkpoint,single_target_checkpoint,checkpoint_list[i]]

    title = 'Multi_Target_Ts_RO_Eval_MA_'+str(cont)
    Charts.create_map_chart(result_path_,labels_,main_path,path_to_export_chart,title,num_samples)

    title = "Metrics_Multi_Target_Ts_RO_Eval_MA_"+str(cont)
    Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,path_to_export_chart,title)
    cont += 1