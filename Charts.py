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
from scipy.interpolate import make_interp_spline
import matplotlib.gridspec as gridspec


colors = ["#dba237", "#70b2e4", "#469b76", "#2932bb", "#d12e95", "#00bfa0", "#ffa300", "#dc0ab4", "#b3d4ff", "#808080"]


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
    
    Hit_map_path = os.path.join(args.average_results_dir, args.file, 'hit_map.npy')
    uncertainty_path = os.path.join(args.average_results_dir, args.file,'Uncertainty_map.npy')
    
    if not os.path.exists(Hit_map_path):
        raise Exception(f"There is no hit map recorded in: {Hit_map_path}")
    
    if not os.path.exists(uncertainty_path):
        raise Exception(f"There is no uncertainty map recorded in: {uncertainty_path}")
    
    uncertainty_map = np.load(uncertainty_path)
    
    Avg_hit_map = np.load(Hit_map_path)

    dataset.Tiles_Configuration(args, 0)
    
    ACCURACY, FSCORE, RECALL, PRECISION, _, _, UNCERTAINTY_MEAN, FSCORE_LOW_UNCERTAINTY, FSCORE_HIGH_UNCERTAINTY, FSCORE_AUDIT, AUDIT_THRESHOLD  = Metrics_For_Test(Avg_hit_map, uncertainty_map, dataset.references[0], dataset.references[1], dataset.Train_tiles, dataset.Valid_tiles, dataset.Undesired_tiles, Thresholds, args)
        
    return ACCURACY[0,0], FSCORE[0,0], RECALL[0,0], PRECISION[0,0], UNCERTAINTY_MEAN[0,0], FSCORE_LOW_UNCERTAINTY[0,0], FSCORE_HIGH_UNCERTAINTY[0,0], FSCORE_AUDIT[0,0], AUDIT_THRESHOLD[0,0]

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
            accuracy,fscore,recall,precision,uncertainty,_,_,_,_ = get_metrics(args)            
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

    plt.figure(figsize=(18,7))
        
    bars_mAP = mAP_list
    #bars_Accuracy = accuracy_list
    bars_F1 = fscore_list
    bars_Recall = recall_list
    bars_Precision = precision_list
    bars_Uncertainty = uncertainty_list

    width = 0.17
       
    align = 'edge'    
    
    bar0 = plt.bar(x - (width*2), bars_mAP, width, label='mAP', color=colors[0], align=align)
    #bar1 = plt.bar(x - (width*1), bars_Accuracy, width, label='Accuracy', color=colors[1], align=align)
    bar2 = plt.bar(x - (width*1), bars_F1, width, label='F1-Score', color=colors[1],align=align)
    #bar3 = plt.bar(x + (width*0), bars_Recall,width, label='Recall', color=colors[2],align=align)
    #bar4 = plt.bar(x + (width*1), bars_Precision,width, label='Precision', color=colors[3],align=align)
    bar5 = plt.bar(x + (width*0), bars_Uncertainty,width, label='Average Uncertainty', color=colors[9],align=align)
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Scores %')
    plt.xlabel('Experiments') 
    plt.title(title,fontsize = 14)  
    rcParams['axes.titlepad'] = 20 
    plt.xticks(x, _experiments)

    plt.bar_label(bar0,fmt='%.1f', padding=4)
    #plt.bar_label(bar1,fmt='%.1f%%', padding=3)
    plt.bar_label(bar2,fmt='%.1f', padding=4)
    #plt.bar_label(bar3,fmt='%.1f', padding=3)
    #plt.bar_label(bar4,fmt='%.1f', padding=3)
    plt.bar_label(bar5,fmt='%.1f', padding=4)

    plt.legend(prop={'size': 14})
    #plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper right')
    plt.legend(bbox_to_anchor=(1, 1.15), loc='upper right')
    
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
    _, ax = plt.subplots(figsize=(20, 8))

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
    uncertainty_title = titles + 'Evaluation of F1-score (%) leveraging uncertainty estimation (Post-Ensemble)\nAudit area = 3%'

    file_title = map_file
    #map_list = create_map_chart(result_path_,labels_,SharedParameters.AVG_MAIN_PATH,SharedParameters.RESULTS_MAIN_PATH,file_title,mapTitle,num_samples,(7,7))
    #create_map_f1_boxplot(result_path_,labels_,SharedParameters.RESULTS_MAIN_PATH, SharedParameters.RESULTS_MAIN_PATH, file_title)

    file_title = metrics_file
    #create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,SharedParameters.RESULTS_MAIN_PATH,file_title,title)
    #create_uncertainty_chart(args,labels_,target,result_path_,checkpoint_list_,SharedParameters.RESULTS_MAIN_PATH,f'{file_title}_Uncertainty', uncertainty_title)
    create_audit_area_chart(baseline_paths, baseline_labels, SharedParameters.RESULTS_MAIN_PATH, f'{file_title}_Audit')
    
    
    
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
                        _,fscore,_,_,_,_,_,_,_ = get_metrics(args)
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
    
def generate_tables_uncertainty(args, sources):
    
    _experiments = [method["name"] for method in sources[0]["targets"][0]["methods"]]
    _sources = [source["name"] for source in sources]
    _targets = [target["name"] for source in sources for target in source["targets"]]
    
    f1_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'fscore_arr_{args.method}.npy')
    f1_low_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'f1_low_arr_{args.method}.npy')
    f1_high_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'f1_high_arr_{args.method}.npy')
    f1_audit_path = os.path.join(SharedParameters.RESULTS_MAIN_PATH,f'f1_audit_arr_{args.method}.npy')
    
    if args.recalculate or not (os.path.exists(f1_path) 
                                and os.path.exists(f1_low_path) 
                                and os.path.exists(f1_high_path) 
                                and os.path.exists(f1_audit_path)):
        
        _f1 = []
        _f1_low = []
        _f1_high = []
        _f1_audit = []
    
        args.save_result_text = True
        
        for source in sources:
            for target in source["targets"]:
                fscore_list = []
                f1_low_list = []
                f1_high_list = []
                f1_audit_list = []
                for method in target["methods"]:
                    args.target_dataset = target['name']
                    args.checkpoint_dir = method['checkpoints']
                    args.results_dir = method['results']
                    
                    #Computing F1-Score
                    try:
                        _,fscore,_,_,_,f1_low,f1_high,f1_audit,_ = get_metrics(args)
                    except Exception as e:
                        print(f"Error: {e}")
                        sys.exit(1)
                    fscore_list.append(fscore)
                    f1_low_list.append(f1_low)
                    f1_high_list.append(f1_high)
                    f1_audit_list.append(f1_audit)
                    
                _f1.append(fscore_list)
                _f1_low.append(f1_low_list)
                _f1_high.append(f1_high_list)
                _f1_audit.append(f1_audit_list)
                
        _f1_arr = np.array(_f1).transpose()
        _f1_low_arr = np.array(_f1_low).transpose()
        _f1_high_arr = np.array(_f1_high).transpose()
        _f1_audit_arr = np.array(_f1_audit).transpose()
        
        np.save(f1_path,_f1_arr)
        np.save(f1_low_path,_f1_low_arr)
        np.save(f1_high_path,_f1_high_arr)
        np.save(f1_audit_path,_f1_audit_arr)
        
    else:
        _f1_arr = np.load(f1_path)
        _f1_low_arr = np.load(f1_low_path)
        _f1_high_arr = np.load(f1_high_path)
        _f1_audit_arr = np.load(f1_audit_path)
    
    
    generate_latex_table_uncertainty(_experiments, _sources, _targets,_f1_arr, _f1_low_arr,_f1_high_arr,_f1_audit_arr, SharedParameters.RESULTS_MAIN_PATH)        
    
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
    
    if map_values.shape[0] > 1 and setting == 'multi-target':
        max_values_map = np.max(map_values[1:], axis=0)
        max_values_f1 = np.max(f1_values[1:], axis=0)
    elif map_values.shape[0] > 1 and setting == 'multi-source':
        max_values_map = np.max(map_values[:], axis=0)
        max_values_f1 = np.max(f1_values[:], axis=0)
    else:
        max_values_map = np.zeros(map_values.shape[1])
        max_values_f1 = np.zeros(map_values.shape[1])
    
    formatted_map_values = np.empty(map_values.shape,dtype=object)
    formatted_f1_values = np.empty(f1_values.shape,dtype=object)
    
    for i in range(map_values.shape[0]):
        for j in range(map_values.shape[1]):
            formatted_map_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{map_values[i, j]:.1f}}}' if map_values[i, j] == max_values_map[j] else f'{map_values[i, j]:.1f}'
            formatted_f1_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{f1_values[i, j]:.1f}}}' if f1_values[i, j] == max_values_f1[j] else f'{f1_values[i, j]:.1f}'
            
    if not multisource:
        latex_table = r"""
        \begin{table}[h!]
        \begingroup
        \setlength{\tabcolsep}{2pt}
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
        latex_table += r"""\end{tabularx}
        \caption{Evaluation of mAP and F1-Score in Multi-Target Experiments.}
        \label{table:""" + f"results_multitarget" + r"""}
        \endgroup
        \end{table}
        """
    else:
        latex_table = r"""
        \begin{table}[h!]
        \begingroup
        \setlength{\tabcolsep}{4pt}
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
        
        latex_table += r"""\end{tabularx}
        \caption{Evaluation of mAP and F1-Score in Multi-Source Experiments.}
        \label{table:""" + f"results_multisource" + r"""}
        \endgroup
        \end{table}
        """

    output_file = os.path.join(file_path, f'{setting}_latex_table.tex')
    
    with open(output_file, 'w') as file:
        file.write(latex_table)
        
def generate_latex_table_uncertainty(experiments, sources, targets, f1_values,f1_low_values,f1_high_values,f1_audit_values, file_path):
    
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
        raise ValueError(f"Mismatch in the lengths of sources({len(sources)}) and targets({len(targets)}).")
        
    if not (len(f1_values) == num_experiments 
            and len(f1_low_values) == num_experiments 
            and len(f1_high_values) == num_experiments 
            and len(f1_audit_values) == num_experiments 
            and len(f1_values[0]) == num_targets 
            and len(f1_low_values[0]) == num_targets 
            and len(f1_high_values[0]) == num_targets 
            and len(f1_audit_values[0]) == num_targets):
        raise ValueError("Mismatch in the lengths of input lists.")
    
    if f1_values.shape[0] > 1 and setting == 'multi-target':
        max_values_f1 = np.max(f1_values[1:], axis=0)
        max_values_f1_low = np.max(f1_low_values[1:], axis=0)
        max_values_f1_high = np.max(f1_high_values[1:], axis=0)
        max_values_f1_audit = np.max(f1_audit_values[1:], axis=0)
    elif f1_values.shape[0] > 1 and setting == 'multi-source':
        max_values_f1 = np.max(f1_values[:], axis=0)
        max_values_f1_low = np.max(f1_low_values[:], axis=0)
        max_values_f1_high = np.max(f1_high_values[:], axis=0)
        max_values_f1_audit = np.max(f1_audit_values[:], axis=0)
    else:
        max_values_f1 = np.zeros(f1_values.shape[1])
        max_values_f1_low = np.zeros(f1_low_values.shape[1])
        max_values_f1_high = np.zeros(f1_high_values.shape[1])
        max_values_f1_audit = np.zeros(f1_audit_values.shape[1])
        
    formatted_f1_values = np.empty(f1_values.shape,dtype=object)
    formatted_f1_low_values = np.empty(f1_low_values.shape,dtype=object)
    formatted_f1_high_values = np.empty(f1_high_values.shape,dtype=object)
    formatted_f1_audit_values = np.empty(f1_audit_values.shape,dtype=object)
    
    for i in range(f1_values.shape[0]):
        for j in range(f1_values.shape[1]):
            formatted_f1_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{f1_values[i, j]:.1f}}}' if f1_values[i, j] == max_values_f1[j] else f'{f1_values[i, j]:.1f}'
            formatted_f1_low_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{f1_low_values[i, j]:.1f}}}' if f1_low_values[i, j] == max_values_f1_low[j] else f'{f1_low_values[i, j]:.1f}'
            formatted_f1_high_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{f1_high_values[i, j]:.1f}}}' if f1_high_values[i, j] == max_values_f1_high[j] else f'{f1_high_values[i, j]:.1f}'
            formatted_f1_audit_values[i,j] = fr'\cellcolor{{lime}}\textbf{{{f1_audit_values[i, j]:.1f}}}' if f1_audit_values[i, j] == max_values_f1_audit[j] else f'{f1_audit_values[i, j]:.1f}'
    
    latex_table = ""      
    if not multisource:
        for index, s in enumerate(sources):
            
            idx = index * 2
            
            latex_table += r"""
            \begin{table}[h!]
            \begingroup
            \setlength{\tabcolsep}{4pt}
            \centering
            \begin{tabularx}{\textwidth}{|c|""" + "CCCC|" * 2 + r"""}
            \hline
            \textbf{Source} & """ + f"\multicolumn{{8}}{{c|}}{{{s}}}" + r""" \\
            \hline
            \textbf{Target} & """ + " & ".join([f"\multicolumn{{4}}{{c|}}{{{targets[idx+j]}}}" for j in range(2)]) + r""" \\
            \hline
            \textbf{Experiments} & """ + " & ".join([f"F1 & F1_{{low}} & F1_{{high}} & F1_{{aud}}" for _ in range(2)]) + r""" \\
            \hline
            """
            for i, experiment in enumerate(experiments):
                latex_table += f"\makecell{{{experiment}}} & " + " & ".join([f"{formatted_f1_values[i,idx+j]} & {formatted_f1_low_values[i,idx+j]} & {formatted_f1_high_values[i,idx+j]} & {formatted_f1_audit_values[i,idx+j]}" for j in range(2)]) + r""" \\
                \hline
                """
            latex_table += r"""\end{tabularx}
            \caption{F1-Score Evaluation for """ + f"{s}" + r""" source in Multi-Target Uncertainty Estimation Experiments.}
            \label{table:""" + f"results_multitarget_uncertainty_{s}" + r"""}
            \endgroup
            \end{table}
            """
    else:
        for index, s in enumerate(sources):
            
            idx = index
            
            latex_table += r"""
            \begin{table}[h!]
            \begingroup
            \setlength{\tabcolsep}{4pt}
            \centering
            \begin{tabularx}{\textwidth}{|c|""" + "CCCC|" + r"""}
            \hline
            \textbf{Source} & """ + f"\multicolumn{{4}}{{c|}}{{{s}}}" + r""" \\
            \hline
            \textbf{Target} & """ + f"\multicolumn{{4}}{{c|}}{{{targets[idx]}}}" + r""" \\
            \hline
            \textbf{Experiments} & F1 & F1_{low} & F1_{high} & F1_{aud} \\
            \hline
            """
            for i, experiment in enumerate(experiments):
                latex_table += f"\makecell{{{experiment}}} & " + " & ".join([f"{formatted_f1_values[i,idx+j]} & {formatted_f1_low_values[i,idx+j]} & {formatted_f1_high_values[i,idx+j]} & {formatted_f1_audit_values[i,idx+j]}" for j in range(1)]) + r""" \\
                \hline
                """
            latex_table += r"""\end{tabularx}
            \caption{F1-Score Evaluation for """ + f"{s}" + r""" source in Multi-Source Uncertainty Estimation Experiments.}
            \label{table:""" + f"results_multisource_uncertainty_{s}" + r"""}
            \endgroup
            \end{table}
            """

    output_file = os.path.join(file_path, f'{setting}_uncertainty_latex_table.tex')
    
    with open(output_file, 'w') as file:
        file.write(latex_table)
        
        
def create_uncertainty_chart(args, experiments, target, result_list, checkpoint_list, path_to_export_chart, file_title, title):
    if not (len(result_list) == len(checkpoint_list) and len(checkpoint_list) == len(experiments)):
        raise Exception("Lists are not the same length. Please verify.")
    
    if len(result_list) != len(set(result_list)):        
        raise Exception("Duplicates found in the result list")
    
    _length = len(result_list)    

    _experiments = []
    fscore_list = []
    fscore_low_list = []
    fscore_high_list = []
    fscore_audit_list = []
    audit_threshold_list = []

    args.save_result_text = True
        
    for i in range(0,_length):
        args.target_dataset = target
        args.checkpoint_dir = checkpoint_list[i]
        args.results_dir = result_list[i]
        try:
            _,fscore,_,_,_,fscore_low_uncertainty, fscore_high_uncertainty, fscore_audit, audit_threshold = get_metrics(args)    
            
            fscore_list.append(fscore)
            fscore_low_list.append(fscore_low_uncertainty) 
            fscore_high_list.append(fscore_high_uncertainty)
            fscore_audit_list.append(fscore_audit)
            audit_threshold_list.append(audit_threshold)
            
        except Exception as e:
            raise e
        
        _experiments.append(experiments[i])

    x = np.arange(len(_experiments))   

    plt.clf()

    plt.figure(figsize=(17,7))
    
    bars_F1 = fscore_list
    bars_F1_Low = fscore_low_list
    bars_F1_High = fscore_high_list
    bars_F1_Audit = fscore_audit_list

    width = 0.17
       
    align = 'edge'    
    
    bar0 = plt.bar(x - (width*2), bars_F1, width, label='F1', color=colors[1], align=align)
    bar1 = plt.bar(x - (width*1), bars_F1_Low, width, label='F1 low uncertainty', color=colors[5],align=align)
    bar2 = plt.bar(x + (width*0), bars_F1_High, width, label='F1 high uncertainty', color=colors[6],align=align)
    bar3 = plt.bar(x + (width*1), bars_F1_Audit, width, label='F1 audit', color=colors[7],align=align)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Scores %')
    plt.xlabel('Experiments') 
    plt.title(title,fontsize = 14)  
    rcParams['axes.titlepad'] = 20 
    plt.xticks(x, _experiments)

    plt.bar_label(bar0,fmt='%.1f', padding=4)
    plt.bar_label(bar1,fmt='%.1f', padding=4)
    plt.bar_label(bar2,fmt='%.1f', padding=4)
    plt.bar_label(bar3,fmt='%.1f', padding=4)

    plt.legend(prop={'size': 14})
    #plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper right')
    plt.legend(bbox_to_anchor=(1, 1.15), loc='upper right')
    
    plt.ylim(0,100)

    full_chart_path = path_to_export_chart + file_title + '.png'

    plt.tight_layout()

    plt.savefig(full_chart_path, format="png")
    plt.close()

    print(f"Done! {full_chart_path} has been saved.")
    
    
def create_audit_area_chart(baseline_paths, baseline_labels, output_directory,filename):
    
    if not (len(baseline_paths) == len(baseline_labels)):
        raise Exception("create_audit_area_chart: Lists are not the same length. Please verify.")
    
    # Plot each column, with column 1 (constant value) as dashed line
    labels = [
        'F1 No Uncertainty',
        'F1 Low Uncertainty',
        'F1 High Uncertainty',
        'F1 Audited'
    ]

    colors = [
        '#5285B9',
        '#70B2E4',
        'red',
        '#63A953'
    ]
    
    row_index = 0
    col_index = 0
 
    # Create a figure
    fig = plt.figure(figsize=(14, 8))

    # Define the GridSpec
    gs = gridspec.GridSpec(3, 3)
    
    for rf in range(len(baseline_paths)):

        if(rf != 0 and rf % 3 == 0):
            row_index+=1
            col_index=0
        
        file_path = os.path.join(SharedParameters.AVG_MAIN_PATH, baseline_paths[rf],'Avg_Scores','fscore_metrics.npy')
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} could not be found.")
        
        fscore_array = np.load(file_path)
        
        data = fscore_array[:,:4]
        # X-axis points (row indices)
        x = np.arange(data.shape[0])

        # Smoothing the lines using cubic spline interpolation
        x_new = np.linspace(x.min(), x.max(), 300)  # More points for a smoother curve

        for i in range(data.shape[1]):
            ax = plt.subplot(gs[row_index, col_index])
            
            spline = make_interp_spline(x, data[:, i], k=3)  # Cubic spline
            y_smooth = spline(x_new)
            
            if np.all(data[:, i] == data[0, i]):  # Check if the column has constant values
                ax.plot(x_new, y_smooth, linestyle='--', label=labels[i], color=colors[i])
            else:
                ax.plot(x_new, y_smooth, label=labels[i], color=colors[i])

        # Adding title and labels
        ax.set_title(f'F1-Score Performance across Audit Areas for {baseline_labels[rf]}')
        ax.set_xlabel('Audit Area (%)')
        ax.set_ylabel('F1 Score (%)')
        ax.set_xticks(np.arange(21))
        ax.set_yticks(np.arange(0,100,10))

        # Adding legend
        ax.legend()
        
        col_index+=1

    # Adjust layout
    plt.tight_layout()        

    full_chart_path = os.path.join(output_directory,filename) + '.png'

    plt.savefig(full_chart_path, format="png")
    plt.close()