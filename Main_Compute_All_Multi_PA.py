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

baseline_paths = [
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/',
    'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/',
    'results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_PA/',
    
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/',
    'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/',
    'results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_PA/',
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/',
    'checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
]

labels = [
    r"Source MA | Target PA | Test PA"+"\n"+r"Source MA | Target PA,RO | Test PA$^{1}$"+"\n"+r"Source MA,RO | Target PA | Test PA$^{2}$",
    r"Source RO | Target PA | Test PA"+"\n"+r"Source RO | Target MA,PA | Test PA$^{1}$"+"\n"+r"Source MA,RO | Target PA | Test PA$^{2}$"
]

'''
labels = [
    r"Source MA | Target PA | Test PA",
    r"Source RO | Target PA | Test PA"
]

compute_f1_audit = [
    False,
    False,
    True,
    
    False,
    False,
    True,
]
'''

compute_f1_audit = [
    False,
    False,
    False,
    False,
    False,
    
    False,
    False,
    False,
    False,
    False,
]

args.checkpoint_results_main_path = "./results/"

target = AMAZON_PA.DATASET

titles = SharedParameters.formatted_f1_chart_title('PA')
metrics_file = f'Metrics_{SharedParameters.DMDA_FILE_TITLE}_PA'

f1Title = 'Evaluation of F1-Score (%) across experiments' + "\n" + titles

Charts.create_f1_bar_chart(args,labels,target,baseline_paths,baseline_checkpoints,SharedParameters.RESULTS_MAIN_PATH,metrics_file,f1Title,compute_f1_audit)

#Charts.create_map_f1_boxplot(baseline_paths,labels,SharedParameters.RESULTS_MAIN_PATH, SharedParameters.RESULTS_MAIN_PATH, boxplot_title)

'''
f1_array, f1_std, f1_mean = Charts.get_stats(baseline_paths,SharedParameters.RESULTS_MAIN_PATH)

print('f1_mean')
print(f1_mean)

print('f1_std')
print(f1_std)

Charts.t_test('Source MA | Target PA -> MT vs Lower baseline',f1_mean[1],f1_std[1],f1_mean[3],f1_std[3], n=5)
Charts.t_test('Source MA | Target PA -> MS vs Lower baseline',f1_mean[1],f1_std[1],f1_mean[4],f1_std[4], n=5)

Charts.t_test('Source RO | Target PA -> MT vs Lower baseline',f1_mean[6],f1_std[6],f1_mean[8],f1_std[8], n=5)
Charts.t_test('Source RO | Target PA -> MS vs Lower baseline',f1_mean[6],f1_std[6],f1_mean[9],f1_std[9], n=5)
'''