import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import sys

from Tools import *
from Models_FC114 import Metrics_For_Test
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters


colors = ["#dba237", "#70b2e4", "#469b76", "#2932bb", "#d12e95", "#00bfa0", "#ffa300", "#dc0ab4", "#b3d4ff"]


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
    
    if args.target_dataset == AMAZON_RO.DATASET or args.target_dataset == 'RO':        
        dataset = AMAZON_RO(args)
    elif args.target_dataset ==  AMAZON_PA.DATASET or args.target_dataset == 'PA':        
        dataset = AMAZON_PA(args)
    elif args.target_dataset == CERRADO_MA.DATASET or args.target_dataset == 'MA':        
        dataset = CERRADO_MA(args)
    else:
        raise Exception("Invalid target_dataset argument: " + args.target_dataset)

    args.area_avoided = int(dataset.AREA_AVOIDED)
    args.horizontal_blocks = int(dataset.HORIZONTAL_BLOCKS)
    args.vertical_blocks = int(dataset.VERTICAL_BLOCKS)
    
    args.average_results_dir = os.path.join(SharedParameters.AVG_MAIN_PATH, args.results_dir)
    args.results_dir = os.path.join(SharedParameters.AVG_MAIN_PATH, args.results_dir)
    args.checkpoint_dir = os.path.join(SharedParameters.CHECKPOINTS_MAIN_PATH, args.checkpoint_dir)
    args.file = 'Avg_Scores'
    args.create_classification_map = True

    if not os.path.exists(args.results_dir):
        raise Exception(f"Folder does not exist: {args.results_dir}")
    if not os.path.exists(args.checkpoint_dir):
        raise Exception(f"Folder does not exist: {args.checkpoint_dir}")
    if not os.path.exists(args.average_results_dir):
        raise Exception(f"Folder does not exist: {args.average_results_dir}")

    ACCURACY_ = []
    FSCORE_ = []
    RECALL_ = []
    PRECISION_ = []    
    
    Hit_map_path = os.path.join(args.average_results_dir, args.file, 'hit_map.npy')
    uncertainty_path = os.path.join(args.average_results_dir, args.file,'Uncertainty_map.npy')
    
    if not os.path.exists(Hit_map_path):
        raise Exception(f"There is no hit map recorded in: {Hit_map_path}")
    
    if not os.path.exists(uncertainty_path):
        raise Exception(f"There is no uncertainty map recorded in: {uncertainty_path}")
    
    uncertainty_map = np.load(uncertainty_path)
    uncertainty_mean = np.mean(uncertainty_map.flatten())*100
    
    Avg_hit_map = np.load(Hit_map_path)

    dataset.Tiles_Configuration(args, 0)
    
    ACCURACY, FSCORE, RECALL, PRECISION, _, _ = Metrics_For_Test(Avg_hit_map, dataset.references[0], dataset.references[1], dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles, Thresholds,args)
    
    ACCURACY_.append(ACCURACY[0,0])
    FSCORE_.append(FSCORE[0,0])
    RECALL_.append(RECALL[0,0])
    PRECISION_.append(PRECISION[0,0])

    ACCURACY_m = np.mean(ACCURACY_)
    FSCORE_m = np.mean(FSCORE_)
    RECALL_m = np.mean(RECALL_)
    PRECISION_m = np.mean(PRECISION_)
    
    return ACCURACY_m,FSCORE_m,RECALL_m,PRECISION_m, uncertainty_mean

def create_chart(args, experiments, target, result_list, checkpoint_list, mAP_list, path_to_export_chart, file_title, title):
    if not (len(result_list) == len(checkpoint_list) and len(checkpoint_list) == len(experiments) and len(experiments)==len(mAP_list)):
        raise Exception("Lists are not the same length. Please verify.")
    
    if len(result_list) != len(set(result_list)):        
        raise Exception("Duplicates found in the result list")
    
    
    _length = len(result_list)    

    _experiments = []
    accuracy_list = []
    fscore_list = []
    recall_list = []
    precision_list = []
    uncertainty_list = []

    args.save_result_text = True
        
    for i in range(0,_length):
        args.target_dataset = target
        args.checkpoint_dir = checkpoint_list[i]
        args.results_dir = result_list[i]
        try:
            accuracy,fscore,recall,precision,uncertainty = get_metrics(args)            
        except Exception as e:
            print("Error:") 
            print(e)
            continue
        
        accuracy_list.append(accuracy)
        fscore_list.append(fscore)
        recall_list.append(recall)
        precision_list.append(precision)
        uncertainty_list.append(uncertainty)
        _experiments.append(experiments[i])    

    x = np.arange(len(_experiments))   

    plt.clf()

    plt.figure(figsize=(14,7))
        
    bars_mAP = mAP_list
    bars_Accuracy = accuracy_list
    bars_F1 = fscore_list
    bars_Recall = recall_list
    bars_Precision = precision_list
    bars_Uncertainty = uncertainty_list

    width = 0.17
       
    align = 'edge'    
    
    bar0 = plt.bar(x - (width*2), bars_mAP, width, label='mAP', color=colors[0], align=align)
    #bar1 = plt.bar(x - (width*1), bars_Accuracy, width, label='Accuracy', color=colors[1], align=align)
    bar2 = plt.bar(x - (width*1), bars_F1, width, label='F1-Score', color=colors[1],align=align)
    bar3 = plt.bar(x + (width*0), bars_Recall,width, label='Recall', color=colors[2],align=align)
    bar4 = plt.bar(x + (width*1), bars_Precision,width, label='Precision', color=colors[3],align=align)
    bar5 = plt.bar(x + (width*2), bars_Uncertainty,width, label='Average Uncertainty', color=colors[4],align=align)
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Scores %')
    plt.xlabel('Experiments') 
    plt.title(title,fontsize = 14)  
    rcParams['axes.titlepad'] = 20 
    plt.xticks(x, _experiments)

    plt.bar_label(bar0,fmt='%.1f', padding=3)
    #plt.bar_label(bar1,fmt='%.1f%%', padding=3)
    plt.bar_label(bar2,fmt='%.1f', padding=3)
    plt.bar_label(bar3,fmt='%.1f', padding=3)
    plt.bar_label(bar4,fmt='%.1f', padding=3)
    plt.bar_label(bar5,fmt='%.1f', padding=3)

    plt.legend(prop={'size': 14})
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper right')
    
    plt.ylim(0,100)

    full_chart_path = path_to_export_chart + file_title + '.png'

    plt.tight_layout()

    plt.savefig(full_chart_path, format="png")
    plt.close()

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

def create_map_chart(result_path,labels,main_path,output_directory,file_title,title,num_samples,chartsize=(7,7), displayChart = True):
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

        MAP = 0

        recall_i = np.zeros((1,num_samples))
        precision_i = np.zeros((1,num_samples))

        AP_i = []
        AP_i_ = 0
        folder_i = os.listdir(result_folder)

        for i in range(len(folder_i)):
            result_folder_name = folder_i[i]
            result_folder_path = os.path.join(result_folder,result_folder_name)

            if result_folder_name != 'Results.txt':
                recall_i, precision_i, mAP = computeMap(num_samples, None, None, result_folder_path)
                print(f"mAP: {mAP:.2f}")
                map_list.append(mAP)
        if displayChart:
            ax.plot(recall_i[0,:], precision_i[0,:], color=colors[rf], label= f'{labels[rf]} - mAP: {mAP:.1f}')

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
        plt.savefig(output_directory + 'Recall_vs_Precision_5_runs_' + file_title + '_DeepLab_Xception.png', format="png")
        plt.close()
        init += 1
    
    return map_list

def computeMap(num_samples,recall, precision, result_folder_path=None):
    Npoints = num_samples
    Interpolation = True
    Correct = True

    if result_folder_path is not None:
        recall_path = os.path.join(result_folder_path, 'Recall.npy')
        precision_path = os.path.join(result_folder_path, 'Precission.npy')
        recall__ = np.load(recall_path)
        precision__ = np.load(precision_path)
    else:
        recall__ = recall
        precision__ = precision

    if np.size(recall__, 1) > Npoints:
        recall__ = recall__[:,:-1]
    if np.size(precision__, 1) > Npoints:
        precision__ = precision__[:,:-1]

    recall__ = recall__/100
    precision__ = precision__/100

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
    
    return recall_i, precision_i, mAP


def create_map_f1_boxplot(result_path,labels,base_path, output_directory, file_title):
    if len(result_path) != len(set(result_path)):        
        raise Exception("Duplicates found in the result list.")
    
    if len(result_path) != len(labels):
        raise Exception("Lists are not the same length. Please verify.")

    map_list = []
    f1_list = []
    
    for rf in range(len(result_path)):

        result_folder = os.path.join(base_path,result_path[rf])

        if not os.path.exists(result_folder):
            continue

        map_values, f1_values = extract_map_and_f1(os.path.join(result_folder,'Results.txt'))

        map_list.append(map_values)
        f1_list.append(f1_values)

    generate_combined_boxplot(output_directory,map_list,'mAP',labels, file_title, f'Evaluation of mAP (%) over 5 Runs (Pre-Ensemble)')
    generate_combined_boxplot(output_directory,f1_list,'F1',labels, file_title, f'Evaluation of F1-Score (%) over 5 Runs (Pre-Ensemble)')


def generate_combined_boxplot(output_directory, data_list, metric, labels, file_title, title):
    """
    Generate combined boxplots for the provided data lists and annotate the mean, upper bound, and lower bound.
    """
    _, ax = plt.subplots(figsize=(16, 8))

    fontsize = 11
    
    # Create the boxplots
    box = ax.boxplot(data_list, vert=True, whis=(0,100), patch_artist=True, labels=labels)

    for median in box['medians']:
        median.set(color ='orange', linewidth = 2)
    
    # Annotate statistics for each boxplot
    for i, data in enumerate(data_list):
        mean_val = np.mean(data)
        lower_bound = np.min(data)
        upper_bound = np.max(data)
        
        # Annotate the mean
        ax.scatter(i + 1, mean_val, color='red', zorder=3)
        ax.text(i + 1.15, mean_val, f'Mean: {mean_val:.1f}', verticalalignment='center', fontsize=fontsize)
        
        # Annotate the lower and upper bounds
        ax.scatter(i + 1, lower_bound, color='blue', zorder=3)
        ax.text(i + 1.15, lower_bound - 1.5, f'Min: {lower_bound:.1f}', verticalalignment='center', fontsize=fontsize)
        
        ax.scatter(i + 1, upper_bound, color='blue', zorder=3)
        ax.text(i + 1.15, upper_bound + 1.5, f'Max: {upper_bound:.1f}', verticalalignment='center', fontsize=fontsize)
    
    # Set plot title and labels
    ax.set_title(title)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_ylim(10, 100)
    ax.grid(True)
    
    plt.savefig(os.path.join(output_directory, f'Boxplot_5_runs_{file_title}_{metric}_DeepLab_Xception.png'), format="png")
    plt.close()

#Usage: 
#map_values, f1_values = extract_map_and_f1('Results.txt')   
def extract_map_and_f1(file_path):
    """
    Extract mAP values from the file.
    
    Parameters:
    - file_path: path to the file containing the results
    
    Returns:
    - A dictionary where keys are experiment identifiers and values are lists of mAP values
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match the experiment identifier and mAP values

    experiment_pattern = re.compile(r"Run: \d+ Accuracy: [\d.]+% F1-Score: ([\d.]+)% Recall: [\d.]+% Precision: [\d.]+% Area: [\d.]+% mAP: ([\d.]+)%")
    
    map_values = []
    f1_values = []
    
    for match in experiment_pattern.findall(content):
        f1_score, map_score = match
        f1_values.append(float(f1_score))
        map_values.append(float(map_score))
    
    return map_values, f1_values


def create_all_charts(args, baseline_paths, baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target):
    result_path_ = baseline_paths
    labels_ = baseline_labels
    checkpoint_list_ = baseline_checkpoints

    title = titles + 'Evaluation of metrics (%) across experiments (Post-Ensemble)'
    mapTitle = titles + 'Evaluation of mAP (%) across experiments (Post-Ensemble)'

    file_title = map_file
    map_list = create_map_chart(result_path_,labels_,SharedParameters.AVG_MAIN_PATH,SharedParameters.RESULTS_MAIN_PATH,file_title,mapTitle,num_samples,(7,7))
    create_map_f1_boxplot(result_path_,labels_,SharedParameters.RESULTS_MAIN_PATH, SharedParameters.RESULTS_MAIN_PATH, file_title)

    file_title = metrics_file
    create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,SharedParameters.RESULTS_MAIN_PATH,file_title,title)
    
def generate_tables(args, sources):
    
    _experiments = [method["name"] for method in sources[0]["targets"][0]["methods"]]
    _sources = [source["name"] for source in sources]
    _targets = [target["name"] for source in sources for target in source["targets"]]
    
    map_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'map_arr_{args.method}.npy')
    f1_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'f1_arr_{args.method}.npy')
    
    if args.recalculate or not (os.path.exists(map_path) and os.path.exists(f1_path)):
        
        _f1 = []
        _mAP = []
    
        args.save_result_text = True
        
        for source in sources:
            for target in source["targets"]:
                fscore_list = []
                mAP_list = []
                for method in target["methods"]:
                    args.target_dataset = target['name']
                    args.checkpoint_dir = method['checkpoints']
                    args.results_dir = method['results']
                    
                    #Computing F1-Score
                    try:
                        _,fscore,_,_,_ = get_metrics(args)
                    except Exception as e:
                        print(f"Error: {e}")
                        sys.exit(1)
                    fscore_list.append(fscore)
                    
                    #Computing mAP
                    avg_result_folder = os.path.join(SharedParameters.AVG_MAIN_PATH, method['results'], 'Avg_Scores')
                    _, _, mAP = computeMap(100, None, None, avg_result_folder)
                    mAP_list.append(mAP)
                _f1.append(fscore_list)
                _mAP.append(mAP_list)
                
        _map_arr = np.array(_mAP).transpose()
        _f1_arr = np.array(_f1).transpose()
        
        np.save(map_path,_map_arr)
        np.save(f1_path,_f1_arr)
        
    else:
        if not (os.path.exists(map_path) and os.path.exists(f1_path)):
            raise Exception(f"Files not found: {map_path} {f1_path}")
        
        _map_arr = np.load(map_path)
        _f1_arr = np.load(f1_path)
    
    generate_latex_table(_experiments, _sources, _targets, _map_arr, _f1_arr, SharedParameters.RESULTS_MAIN_PATH)        
    
def generate_latex_table(experiments, sources, targets, map_values, f1_values, file_path):
    
    # Verify the lengths of the inputs
    num_experiments = len(experiments)
    num_targets = len(targets)
    multisource = False
    setting = 'multi-target'

    #Multisource cenario example: source=['PA-MA','PA-RO','MA-RO'] target= ['RO','MA','PA']
    if len(sources) == len(targets):
        multisource = True
        setting = 'multi-source'
    elif len(sources) * 2 != len(targets):
        raise ValueError("Mismatch in the lengths of sources and targets.")
        
    if not (len(map_values) == num_experiments and len(f1_values) == num_experiments and len(map_values[0]) == num_targets and len(f1_values[0]) == num_targets):
        raise ValueError("Mismatch in the lengths of input lists.")
    
    if map_values.shape[0] > 1:
        max_values_map = np.max(map_values[1:], axis=0)
        max_values_f1 = np.max(f1_values[1:], axis=0)
    else:
        max_values_map = np.zeros(map_values.shape[1])
        max_values_f1 = np.zeros(map_values.shape[1])
    
    formatted_map_values = np.empty(map_values.shape,dtype=object)
    formatted_f1_values = np.empty(f1_values.shape,dtype=object)
    
    for i in range(map_values.shape[0]):
        for j in range(map_values.shape[1]):
            formatted_map_values[i,j] = fr'\textbf{{{map_values[i, j]:.1f}}}' if map_values[i, j] == max_values_map[j] else f'{map_values[i, j]:.1f}'
            formatted_f1_values[i,j] = fr'\textbf{{{f1_values[i, j]:.1f}}}' if f1_values[i, j] == max_values_f1[j] else f'{f1_values[i, j]:.1f}'
            
    if not multisource:
        latex_table = r"""
        \begin{table}[h!]
        \centering
        \begin{tabularx}{\textwidth}{|c|""" + "CC|" * len(targets) + r"""}
        \hline
        \textbf{Source} & """ + " & ".join([f"\multicolumn{{4}}{{c|}}{{{s}}}" for s in sources]) + r""" \\
        \hline
        \textbf{Target} & """ + " & ".join([f"\multicolumn{{2}}{{c|}}{{{t}}}" for t in targets]) + r""" \\
        \hline
        \textbf{Experiments} & """ + " & ".join([f"mAP & F1" for _ in targets]) + r""" \\
        \hline
        """
        for i, experiment in enumerate(experiments):
            
            latex_table += f"\makecell{{{experiment}}} & " + " & ".join([f"{formatted_map_values[i,j]} & {formatted_f1_values[i,j]}" for j in range(len(targets))]) + r""" \\
            \hline
            """
        latex_table += r"""
        \end{tabularx}
        \caption{mAP and F1-Score values for different experiments and combinations of source and targets.}
        \label{table:results}
        \end{table}
        """
    else:
        latex_table = r"""
        \begin{table}[h!]
        \centering
        \begin{tabularx}{\textwidth}{|c|""" + "CC|" * len(targets) + r"""}
        \hline
        \textbf{Source} & """ + " & ".join([f"\multicolumn{{2}}{{c|}}{{{s}}}" for s in sources]) + r""" \\
        \hline
        \textbf{Target} & """ + " & ".join([f"\multicolumn{{2}}{{c|}}{{{t}}}" for t in targets]) + r""" \\
        \hline
        \textbf{Experiments} & """ + " & ".join([f"mAP & F1" for _ in targets]) + r""" \\
        \hline
        """
        
        for i, experiment in enumerate(experiments):
            latex_table += f"\makecell{{{experiment}}} & " + " & ".join([f"{formatted_map_values[i,j]} & {formatted_f1_values[i,j]}" for j in range(len(targets))]) + r""" \\
            \hline
            """
        
        latex_table += r"""
        \end{tabularx}
        \caption{mAP and F1-Score values for different experiments and combinations of source and targets.}
        \label{table:results}
        \end{table}
        """

    output_file = os.path.join(file_path, f'{setting}_latex_table.tex')
    
    with open(output_file, 'w') as file:
        file.write(latex_table)