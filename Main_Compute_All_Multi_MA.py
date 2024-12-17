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
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/',
    'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/',
    'results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/',
    'results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
]

baseline_checkpoints = [
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
]

labels = [
    'Source PA | Target MA',
    'Source RO | Target MA'
]

baseline_labels = []

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = SharedParameters.RESULTS_MAIN_PATH

target = CERRADO_MA.DATASET

titles = SharedParameters.formatted_f1_chart_title('MA')
metrics_file = f'Metrics_{SharedParameters.DMDA_FILE_TITLE}_MA'

f1Title = 'Evaluation of F1-Score (%) across experiments' + "\n" + titles

Charts.create_f1_bar_chart(args,labels,target,baseline_paths,baseline_checkpoints,SharedParameters.RESULTS_MAIN_PATH,metrics_file,f1Title)