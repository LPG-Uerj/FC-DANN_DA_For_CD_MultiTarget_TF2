import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from Tools import *
from Models_FC114 import Metrics_For_Test
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

#colors = []
#colors.append('#1767BD')
#colors.append('#FA7703')
#colors.append('#1AB023')
#colors.append('#7D8080')
#colors.append('#E7E424')
#colors.append('#7125B0')
#colors.append('#2FE5EB')
#colors.append('#FF0000')

colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]




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

            args.create_classification_map = False
            
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

def create_chart(args, experiments, target, result_list,checkpoint_list, mAP_list, path_to_export_chart, file_title, title):
    if not (len(result_list) == len(checkpoint_list) and len(checkpoint_list) == len(experiments) and len(experiments)==len(mAP_list)):
        raise Exception("Lists are not the same length. Please verify.")
    
    if len(result_list) != len(set(result_list)):        
        raise Exception("Duplicates found in the result list")
    
    #if len(experiments) != len(set(experiments)):        
    #    raise Exception("Duplicates found in the experiment list")
    
    _length = len(result_list)    

    _experiments = []
    accuracy_list = []
    fscore_list = []
    recall_list = []
    precision_list = []

    args.save_result_text = True
    temp_metrics = os.path.join(path_to_export_chart,'temp')

    if not os.path.exists(temp_metrics):
        os.makedirs(temp_metrics)
        
    for i in range(0,_length):
        args.target_dataset = target
        args.checkpoint_dir = checkpoint_list[i]
        args.results_dir = result_list[i]
        try:
            accuracy,fscore,recall,precision = get_metrics(args)            
        except Exception as e:
            print("Error:") 
            print(e)
            continue
        
        accuracy_list.append(round(float(accuracy), 1))
        fscore_list.append(round(float(fscore), 1))
        recall_list.append(round(float(recall), 1))
        precision_list.append(round(float(precision), 1))
        _experiments.append(experiments[i])    

    x = np.arange(len(_experiments))   

    plt.clf()

    plt.figure(figsize=(14,7))
        
    bars_mAP = mAP_list
    bars_Accuracy = accuracy_list
    bars_F1 = fscore_list
    bars_Recall = recall_list
    bars_Precision = precision_list

    width = 0.17
       
    align = 'edge'    
    
    bar0 = plt.bar(x - (width*2), bars_mAP, width, label='mAP', color=colors[0], align=align)
    bar1 = plt.bar(x - (width*1), bars_Accuracy, width, label='Accuracy', color=colors[1], align=align)
    bar2 = plt.bar(x + (width*0), bars_F1, width, label='F1-Score', color=colors[2],align=align)
    bar3 = plt.bar(x + (width*1), bars_Recall,width, label='Recall', color=colors[3],align=align)
    bar4 = plt.bar(x + (width*2), bars_Precision,width, label='Precision', color=colors[4],align=align)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Scores %')
    plt.xlabel('Experiments') 
    plt.title(title,fontsize = 14)  
    rcParams['axes.titlepad'] = 20 
    plt.xticks(x, _experiments)

    plt.bar_label(bar0,fmt='%.1f%%', padding=3)
    plt.bar_label(bar1,fmt='%.1f%%', padding=3)
    plt.bar_label(bar2,fmt='%.1f%%', padding=3)
    plt.bar_label(bar3,fmt='%.1f%%', padding=3)
    plt.bar_label(bar4,fmt='%.1f%%', padding=3)

    plt.legend(prop={'size': 14})
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper right')
    
    plt.ylim(0,100)

    full_chart_path = path_to_export_chart + file_title + '.png'

    plt.tight_layout()

    plt.savefig(full_chart_path, format="png")

    print(f"Done! {full_chart_path} has been saved.")

def Area_under_the_curve(X, Y):
    X = X[0,:]
    Y = Y[0,:]
    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])

    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))

    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))

    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)

    return area

def create_map_chart(result_path,labels,main_path,path_to_export_chart,file_title,title,num_samples,chartsize=(7,7), displayChart = True):
    if len(result_path) != len(set(result_path)):        
        raise Exception("Duplicates found in the result list.")
    
    if len(result_path) != len(labels):
        raise Exception("Lists are not the same length. Please verify.")

    init = 0    
    results_folders = result_path
    
    if displayChart:
        plt.figure(figsize=chartsize)
        ax = plt.subplot(111)
    
    map_list = []

    Npoints = num_samples
    Interpolation = True
    Correct = True
    for rf in range(len(results_folders)):

        result_folder = os.path.join(main_path,results_folders[rf])

        if not os.path.exists(result_folder):
            continue

        recall = np.zeros((1 , num_samples))
        precision = np.zeros((1 , num_samples))

        MAP = 0

        recall_i = np.zeros((1,num_samples))
        precision_i = np.zeros((1,num_samples))

        AP_i = []
        AP_i_ = 0
        folder_i = os.listdir(result_folder)

        for i in range(len(folder_i)):
            result_folder_name = folder_i[i]
            if result_folder_name != 'Results.txt':
                #print(folder_i[i])
                recall_path = result_folder + folder_i[i] + '/Recall.npy'
                precision_path = result_folder + folder_i[i] + '/Precission.npy'
                fscore_path = result_folder + folder_i[i] + '/Fscore.npy'

                recall__ = np.load(recall_path)
                precision__ = np.load(precision_path)
                fscore__ = np.load(fscore_path)

                print(precision__)

                if np.size(recall__, 1) > Npoints:
                    recall__ = recall__[:,:-1]
                if np.size(precision__, 1) > Npoints:
                    precision__ = precision__[:,:-1]

                recall__ = recall__/100
                precision__ = precision__/100

                print()

                if Correct:

                    if precision__[0 , 0] == 0:
                        precision__[0 , 0] = 2 * precision__[0 , 1] - precision__[0 , 2]

                    if Interpolation:
                        precision = precision__[0,:]
                        precision__[0,:] = np.maximum.accumulate(precision[::-1])[::-1]


                    if recall__[0 , 0] > 0:
                        recall = np.zeros((1,num_samples + 1))
                        precision = np.zeros((1,num_samples + 1))
                        # Replicating precision value
                        precision[0 , 0] = precision__[0 , 0]
                        precision[0 , 1:] = precision__
                        precision__ = precision
                        # Attending recall
                        recall[0 , 1:] = recall__
                        recall__ = recall

                recall_i = recall__
                precision_i = precision__

                mAP = Area_under_the_curve(recall__, precision__)
                print(mAP)
        map_list.append(np.round(mAP,1))
        if displayChart:
            ax.plot(recall_i[0,:], precision_i[0,:], color=colors[rf], label=labels[rf] + ' - mAP:' + str(np.round(mAP,1)))

    if displayChart:
        ax.legend(prop={'size': 14})

        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.grid(True)
        plt.title(title,fontsize = 14)
        plt.margins(0.2)
        plt.tight_layout(pad=2.0)
        ##rcParams['axes.titlepad'] = 10 
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(path_to_export_chart + 'Recall_vs_Precision_5_runs_' + file_title + '_DeepLab_Xception.png', format="png")
        init += 1
    
    return map_list