import argparse
import Charts

parser = argparse.ArgumentParser(description='')

parser.add_argument('--source_dataset', dest='source_dataset', type=str, default=None,help='The name of the original source dataset used')
parser.add_argument('--target_dataset', dest='target_dataset', type=str, default=None,help='The name of the target dataset whose metrics will be computed')
args = parser.parse_args()

def main():

    if args.source_dataset == 'Amazon_RO':   
        experiments = ['Tr: RO, Ts: RO'
                    #'Tr: RO->PA,MA, Ts: PA',
                    #'Tr: RO->PA,MA, Ts: MA',
                    #'Tr: RO->PA,MA, Ts: RO'
                    ]
    elif args.source_dataset == 'Amazon_PA':
        experiments = ['Tr: PA, Ts: PA'
                    #'Tr: PA->RO,MA, Ts: RO',
                    #'Tr: PA->RO,MA, Ts: MA',
                    #'Tr: PA->RO,MA, Ts: PA'
                    ]
    elif args.source_dataset == 'Cerrado_MA':
        experiments = ['Tr: MA, Ts: MA'
                    #'Tr: MA->RO,PA, Ts: RO',
                    #'Tr: MA->RO,PA, Ts: PA',
                    #'Tr: MA->RO,PA, Ts: MA'
                    ]
    else:
        raise Exception("Invalid argument source_dataset: " + args.source_dataset) 

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision
    title = "Metrics_Source_"+args.source_dataset

    main_path = "./results/"
    path_to_export_chart = "./results/results/"

    result_list = [main_path + 'results_tr_Amazon_RO_DeepLab_None_Amazon_RO_multi_Amazon_RO/']
    checkpoint_list = [main_path + 'results_tr_Amazon_RO_DeepLab_None_Amazon_RO_multi_Amazon_RO/']    

    Charts.create_chart(experiments,result_list,checkpoint_list,path_to_export_chart,title)

if __name__=='__main__':
    main()