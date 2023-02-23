import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Charts


num_samples = 100
result_path = []

main_path = "./results/results_avg/"

result_path = [
    main_path + 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
    main_path + 'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_RO/',
    main_path + 'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_PA/',
    main_path + 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/']

labels = [
    '1-Tr: PA,Ts: PA (Source only training)',
    '2-Tr: RO->PA, Ts: PA (domain adaptation single-target)',
    '3-Tr: RO->PA, Ts: RO (domain adaptation single-target)',
    '4-Tr: PA->RO, Ts: PA (domain adaptation single-target)',
    '5-Tr: RO,Ts: PA (Source only training)']

title = 'Source_RO_Target_PA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)