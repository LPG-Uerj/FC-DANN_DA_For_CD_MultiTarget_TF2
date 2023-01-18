import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Charts


num_samples = 100
result_path = []

main_path = "./results/results_avg/"


#X = RO, Y = RO
result_path.append(main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/')


labels = []
labels.append('1-Tr: RO,Ts: RO (Source only training)')


colors = Charts.colors.copy()

titles = []

titles.append('Target = RO')

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

if __name__ == '__main__':
    init = 0
    for i in range(0, len(result_path), 4):
        results_folders = result_path[i : i + 4]

        fig = plt.figure()
        ax = plt.subplot(111)
        Npoints = num_samples
        Interpolation = True
        Correct = True
        for rf in range(len(results_folders)):

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
            ax.plot(recall_i[0,:], precision_i[0,:], color=colors[rf], label=labels[rf] + 'mAP:' + str(np.round(mAP,1)))

        ax.legend()

        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.grid(True)
        plt.title(titles[init])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(main_path + 'Recall_vs_Precision_5_runs_Target_RO_DeepLab_Xception.png')
        init += 1
