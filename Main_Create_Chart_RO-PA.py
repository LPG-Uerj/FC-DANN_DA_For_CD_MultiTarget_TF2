import argparse
import Charts
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

parser = argparse.ArgumentParser(description='')

parser.add_argument('--debugMode',dest='debugMode', type=eval, choices=[True, False], default=False)

args = parser.parse_args()

def main():
    
    experiments = [
        'Tr:PA-Ts:PA',                        
        'Tr:RO->PA-Ts:PA',    
        'Tr:RO->PA,RO(blcd)-\nTs:PA(Upper)',       
        'Tr:RO->PA,RO-Ts:PA',    
        'Tr:RO->PA,RO(blcd)-\nTs:PA',        
        'Tr:RO-Ts:PA'
        ]

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision
    title = "Metrics_Tr_RO_Ts_PA"
    
    path_to_export_chart = "./results/results/"
    #path_to_export_chart = "../"

   
    result_list = [
        'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',        
        'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA/',
        'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/',
        'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/'      
        ]
    checkpoint_list = [
        'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',        
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_classification_Amazon_RO/'
        ] 
    target_list = [
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET]

    Charts.create_chart(args,experiments,target_list,result_list,checkpoint_list,path_to_export_chart,title)

if __name__=='__main__':
    main()