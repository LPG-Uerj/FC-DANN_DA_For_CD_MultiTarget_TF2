import os
import numpy as np
import matplotlib.pyplot as plt

from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

colors = []
colors.append('#1724BD')
colors.append('#0EB7C2')
colors.append('#BF114B')
colors.append('#E98E2C')
colors.append('#008f39')
colors.append('#663300')

def get_metrics(args):
    Thresholds = np.array([0.5])
    
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
    counter = 0
    files = os.listdir(args.results_dir)

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

def create_chart(experiments, result_list,checkpoint_list, path_to_export_chart, title, args):
    args.save_result_text = True
    for i in range(0,len(result_list)):
        args.checkpoint_dir = checkpoint_list[i]
        args.results_dir = result_list[i]
        accuracy,fscore,recall,precision = get_metrics(args)

    x = np.arange(len(experiments))
    # set width of bars
    barWidth = 0.25
    
    bars_Accuracy = accuracy.copy()
    bars_F1 = fscore.copy()
    bars_Recall = recall.copy()
    bars_Precision = precision.copy()

    width = 0.35
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars_Accuracy))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]


    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bars_Accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, bars_F1, width, label='F1-Score')
    rects3 = ax.bar(x + width/2, bars_Recall, width, label='Recall')
    rects4 = ax.bar(x + width/2, bars_Precision, width, label='Precision')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores %')
    ax.set_xlabel('Experiments') 
    ax.set_title(title)   
    ax.set_xticks(x, experiments)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    ax.set_ylim(0,100)

    fig.tight_layout()  

    plt.show()  

    plt.savefig(path_to_export_chart + title + '.png')