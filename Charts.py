import os
import numpy as np
import matplotlib.pyplot as plt

from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

colors = []
colors.append('#1724BD')
colors.append('#0EB7C2')
colors.append('#BF114B')
colors.append('#E98E2C')
colors.append('#008f39')
colors.append('#663300')

def get_metrics(args):
    Thresholds = np.array([0.5])

    args.checkpoint_results_main_path = "./results/"
    args.eliminate_regions = True
    args.defined_before = False
    args.reference_t2_name = None
    args.fixed_tiles = True
    args.patches_dimension = 64
    args.compute_ndvi = False
    args.image_channels = 7
    args.phase = SharedParameters.PHASE_METRICS
    args.buffer = True
    
    if args.target_dataset == AMAZON_RO.DATASET:        
        dataset = AMAZON_RO(args)
    elif args.target_dataset ==  AMAZON_PA.DATASET:        
        dataset = AMAZON_PA(args)
    elif args.target_dataset == CERRADO_MA.DATASET:        
        dataset = CERRADO_MA(args)
    else:
        raise Exception("Invalid target_dataset argument: " + args.target_dataset)

    args.area_avoided = int(dataset.AREA_AVOIDED)
    args.horizontal_blocks = int(dataset.HORIZONTAL_BLOCKS)
    args.vertical_blocks = int(dataset.VERTICAL_BLOCKS)

    args.results_dir = args.checkpoint_results_main_path + 'results/' + args.results_dir + '/'
    args.checkpoint_dir = args.checkpoint_results_main_path + 'checkpoints/' + args.checkpoint_dir + '/'

    if not os.path.exists(args.results_dir):
        raise Exception(f"Folder does not exist: {args.results_dir}")
    if not os.path.exists(args.checkpoint_dir):
        raise Exception(f"Folder does not exist: {args.checkpoint_dir}")


    counter = 0
    files = os.listdir(args.results_dir)

    if len(files) == 0:
        raise Exception(f"There is no result recorded in: {args.results_dir}")

    ACCURACY_ = []
    FSCORE_ = []
    RECALL_ = []
    PRECISION_ = []    

    for i in range(0, len(files)):
        Hit_map_path = args.results_dir + files[i] + '/hit_map.npy'
        args.file = files[i]
        if os.path.exists(Hit_map_path):
            hit_map = np.load(Hit_map_path)
            fields_file = files[i].split('_')
            checkpoint_name = fields_file[0] + '_' + fields_file[3] + '_' + fields_file[1] + '_' + fields_file[4] + '_' + fields_file[5] + '_' + fields_file[6] + '_' + fields_file[7] + '_'+ fields_file[8] + '_' + fields_file[9] + '_' + fields_file[10] + '_' + fields_file[11]
            args.save_checkpoint_path = args.checkpoint_dir + '/' + checkpoint_name + '/'
            #need to put the path of the checkpoint to recover if needed the original train, validation, and test tiles.
            dataset.Tiles_Configuration(args, i)
            ACCURACY, FSCORE, RECALL, PRECISION, _, _ = Metrics_For_Test(hit_map, dataset.references[0], dataset.references[1], dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles,Thresholds,args)
            ACCURACY_.append(ACCURACY[0,0])
            FSCORE_.append(FSCORE[0,0])
            RECALL_.append(RECALL[0,0])
            PRECISION_.append(PRECISION[0,0])           

            counter += 1
    
    ACCURACY_m = np.mean(ACCURACY_)
    FSCORE_m = np.mean(FSCORE_)
    RECALL_m = np.mean(RECALL_)
    PRECISION_m = np.mean(PRECISION_)
    
    return ACCURACY_m,FSCORE_m,RECALL_m,PRECISION_m

def create_chart(args,experiments, target_list, result_list,checkpoint_list, path_to_export_chart, title):
    if not (len(target_list) == len(result_list) and len(target_list) == len(checkpoint_list)):
        raise Exception("Lists are not the same length. Please verify.")
    
    _experiments = []
    accuracy_list = []
    fscore_list = []
    recall_list = []
    precision_list = []
    args.save_result_text = True
    for i in range(0,len(result_list)):
        args.target_dataset = target_list[i]
        args.checkpoint_dir = checkpoint_list[i]
        args.results_dir = result_list[i]
        try:
            accuracy,fscore,recall,precision = get_metrics(args)
        except Exception as e: 
            print(e)
            continue
        accuracy_list.append("%.2f"%(accuracy))
        fscore_list.append("%.2f%%"%(fscore))
        recall_list.append("%.2f"%(recall))
        precision_list.append("%.2f"%(precision))
        _experiments.append(experiments[i])

    x = np.arange(len(_experiments))
        
    bars_Accuracy = accuracy_list.copy()
    bars_F1 = fscore_list.copy()
    bars_Recall = recall_list.copy()
    bars_Precision = precision_list.copy()

    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, bars_Accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width, bars_F1, width, label='F1-Score')
    rects3 = ax.bar(x + 2*(width), bars_Recall, width, label='Recall')
    rects4 = ax.bar(x + 3*(width), bars_Precision, width, label='Precision')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores %')
    ax.set_xlabel('Experiments') 
    ax.set_title(title)   
    ax.set_xticks(x, _experiments)
    ax.legend()

    ax.bar_label(rects1, padding=4)
    ax.bar_label(rects2, padding=4)
    ax.bar_label(rects3, padding=4)
    ax.bar_label(rects4, padding=4)

    ax.set_ylim(0,100)

    fig.tight_layout()  

    plt.show()  

    full_chart_path = path_to_export_chart + title + '.png'

    plt.savefig(full_chart_path)

    print(f"Done! {full_chart_path} has been saved.")