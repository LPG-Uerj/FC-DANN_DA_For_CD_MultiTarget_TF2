import Charts
import argparse
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

parser = argparse.ArgumentParser()

parser.add_argument('--mapchart', dest='mapchart', type=eval, choices=[True, False], default=True)
parser.add_argument('--f1chart', dest='f1chart', type=eval, choices=[True, False], default=True)

args = parser.parse_args()

num_samples = 100
result_path = []

main_path = "./results/results_avg/"

lower_bound_path = 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/'
lower_bound_checkpoint = 'checkpoint_tr_Amazon_PA_classification_Amazon_PA/'

upper_bound_source_only_path = 'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/'
upper_bound_source_only_checkpoint = 'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/'

single_target_path = 'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/'
single_target_checkpoint = 'checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/'



result_path = [    
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/',
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Cerrado_MA_skipconn_True/',
    #'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_gamma_0.25_Cerrado_MA_skipconn_True/'
]

checkpoint_list = [
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    #'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_gamma_0.25_Amazon_RO_Cerrado_MA_skipconn_True/'
]

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = "./results/results/"

source = AMAZON_PA.DATASET
target = CERRADO_MA.DATASET
cont = 1

if len(result_path) == 0:
    titles = 'X=PA, Y=MA->PA\n'
    map_file = 'mAP_Ts_PA_Eval_MA_'
    metrics_file = 'metrics_Ts_PA_Eval_MA_'
    if SharedParameters.INCLUDE_DA_RESULTS:
        titles = 'X=PA, Y=MA->PA\n'
        result_path_ = [upper_bound_source_only_path,lower_bound_path,single_target_path]
        labels_ = [SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,SharedParameters.LOWER_BOUND_LABEL,SharedParameters.SINGLE_TARGET_LABEL]
        checkpoint_list_ = [upper_bound_source_only_checkpoint,lower_bound_checkpoint,single_target_checkpoint]
        title = titles + "DA single-target {0}-{1}".format(source,target)
        map_file = map_file + "_single"
        metrics_file = metrics_file + "_single"
    else:
        titles = 'X=PA, Y=MA\n'
        result_path_ = [upper_bound_source_only_path,lower_bound_path]
        labels_ = [SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,SharedParameters.LOWER_BOUND_LABEL]
        checkpoint_list_ = [upper_bound_source_only_checkpoint,lower_bound_checkpoint]
        title = titles + "Baseline {0}-{1}".format(source,target)
    if args.mapchart:   
        file_title = map_file+str(cont)        
        map_list = Charts.create_map_chart(result_path_,labels_,SharedParameters.AVG_MAIN_PATH,path_to_export_chart,file_title,title,num_samples,(7,7))
    if args.f1chart:
        file_title = metrics_file+str(cont)        
        Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,path_to_export_chart,file_title,title)
else:
    titles = 'X=PA, Y=MA(Y1=RO,Y2=MA)\n'
    map_file = 'Multi_Target_Ts_PA_Eval_MA_'
    metrics_file = 'Metrics_Multi_Target_Ts_PA_Eval_MA_'
    for i in range(0, len(result_path)):
        result_path_ = [upper_bound_source_only_path,lower_bound_path,single_target_path,result_path[i]]
        labels_ = [SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,SharedParameters.LOWER_BOUND_LABEL,SharedParameters.SINGLE_TARGET_LABEL,SharedParameters.MULTI_TARGET_LABEL]
        checkpoint_list_ = [upper_bound_source_only_checkpoint,lower_bound_checkpoint,single_target_checkpoint,checkpoint_list[i]]

        title = titles + "DA multi-target " + SharedParameters.EXPERIMENTS_LABELS[i]
        if args.mapchart:   
            file_title = map_file+str(cont)        
            map_list = Charts.create_map_chart(result_path_,labels_,SharedParameters.AVG_MAIN_PATH,path_to_export_chart,file_title,title,num_samples,(7,7))
            Charts.create_map_f1_boxplot(result_path_,labels_,SharedParameters.RESULTS_MAIN_PATH, path_to_export_chart, file_title)
        if args.f1chart:
            file_title = metrics_file+str(cont)        
            Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,path_to_export_chart,file_title,title)
        cont += 1

    '''
    title = titles + SharedParameters.DA_CHART_TITLE
    if args.mapchart:   
        file_title = map_file+str(cont)    
        map_list = Charts.create_map_chart(result_path,SharedParameters.EXPERIMENTS_LABELS_LB,main_path,path_to_export_chart,file_title,title,num_samples,(7,7))
    if args.f1chart:    
        file_title = metrics_file+str(cont)  
        Charts.create_chart(args,SharedParameters.EXPERIMENTS_LABELS_LB,target,result_path,checkpoint_list,map_list,path_to_export_chart,file_title,title)\
    '''