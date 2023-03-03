import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Charts


num_samples = 100

main_path = "./results/results_avg/"

result_path = [
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Amazon_RO/',
    'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Cerrado_MA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/']


labels = [
    '1-Tr: MA,Ts: MA (Source only training)',
    '2-Tr: RO->MA, Ts: MA (domain adaptation single-target)',
    '3-Tr: RO->MA, Ts: RO (domain adaptation single-target)',
    '4-Tr: MA->RO, Ts: MA (domain adaptation single-target)',
    '5-Tr: RO, Ts: MA (Source only training)']

title = 'Source_RO_Target_MA'

Charts.create_map_chart(result_path,labels,main_path,title,num_samples)