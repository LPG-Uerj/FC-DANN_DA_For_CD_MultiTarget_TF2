import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Charts


num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = PA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/')

#X = RO->PA, Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/')

#X = PA->RO, Y = PA
result_path.append(main_path + 'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_PA/')

#X = RO, Y = PA
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/')



labels = []
labels.append('1-Tr: PA,Ts: PA (Source only training)')
labels.append('2-Tr: RO->PA, Ts: PA (domain adaptation single-target)')
labels.append('3-Tr: PA->RO, Ts: PA (domain adaptation single-target)')
labels.append('4-Tr: RO,Ts: PA (Source only training)')

colors = Charts.colors.copy()

title = 'Source_RO_Target_PA'

if __name__ == '__main__':
    init = 0
    #for i in range(0, len(result_path), 4):
    #   results_folders = result_path[i : i + 4]
    results_folders = result_path

    fig = plt.figure()
    ax = plt.subplot(111)
    Npoints = num_samples
    Interpolation = True
    Correct = True
    for rf in range(len(results_folders)):

        if not os.path.exists(results_folders[rf]):
            continue

        recall = np.zeros((1 , num_samples))
        precision = np.zeros((1 , num_samples))

        MAP = 0

        recall_i = np.zeros((1,num_samples))
        precision_i = np.zeros((1,num_samples))

        AP_i = []
        AP_i_ = 0
        folder_i = os.listdir(results_folders[rf])

        for i in range(len(folder_i)):
            result_folder_name = folder_i[i]
            if result_folder_name != 'Results.txt':
                #print(folder_i[i])
                recall_path = results_folders[rf] + folder_i[i] + '/Recall.npy'
                precision_path = results_folders[rf] + folder_i[i] + '/Precission.npy'
                fscore_path = results_folders[rf] + folder_i[i] + '/Fscore.npy'

                recall__ = np.load(recall_path)
                precision__ = np.load(precision_path)
                fscore__ = np.load(fscore_path)

                print(precision__)

                #print(precision__)
                #print(recall__)

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

                mAP = Charts.Area_under_the_curve(recall__, precision__)
                print(mAP)
        ax.plot(recall_i[0,:], precision_i[0,:], color=colors[rf], label=labels[rf] + 'mAP:' + str(np.round(mAP,1)))

    ax.legend()

    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.title(title)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(main_path + 'Recall_vs_Precision_5_runs_'+title+'_DeepLab_Xception.png')
    init += 1