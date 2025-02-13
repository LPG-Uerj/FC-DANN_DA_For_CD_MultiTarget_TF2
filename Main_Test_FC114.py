import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import skimage.morphology
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.morphology import square, disk
from sklearn.preprocessing import StandardScaler
from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts
# Model
parser.add_argument('--classifier_type', dest='classifier_type', type=str, default='DeepLab', help='method that will be used, could be used also (siamese_network)')
parser.add_argument('--skip_connections', dest='skip_connections', type=eval, choices=[True, False], default=False, help='method that will be used, could be used also (siamese_network)')
parser.add_argument('--domain_regressor_type', dest='domain_regressor_type', type=str, default='FC', help='Architecture of Domain regressor. Values:FC|CONV')
parser.add_argument('--DR_Localization', dest='DR_Localization', type=int, default=-1, help='The layer in whic the Domain regressor will act')
# Testing parameters
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4000, help='number images in batch')
parser.add_argument('--vertical_blocks', dest='vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--horizontal_blocks', dest='horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--overlap', dest='overlap', type=float, default= 0.75, help= 'stride cadence')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=64, help= 'dimension of the extracted patches')
parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=False, help='Compute and stack the ndvi index to the rest of bands')
parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=False, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=2, help='Number of classes comprised in both domains (classification)')

# Phase
parser.add_argument('--phase', dest='phase', default='test', help='train, test, generate_image, create_dataset')
parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation')
#Checkpoint dir
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./DA_prove', help='Domain adaptation checkpoints')
parser.add_argument('--results_dir', dest='results_dir', type=str, default='./results_DA_prove', help='results will be saved here')
parser.add_argument('--da_type', dest='da_type', type=str, default='CL', help='CL|DR|CL_DR')
# Images dir and names
# Images dir and names
parser.add_argument('--dataset', dest='dataset', type=str, default='Amazonia_Legal/',help='The name of the dataset used')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')
parser.add_argument('--data_t1_year', dest='data_t1_year', type=str, default='2016', help='Year of the time 1 image')
parser.add_argument('--data_t2_year', dest='data_t2_year', type=str, default='2017', help='Year of the time 2 image')
parser.add_argument('--data_t1_name', dest='data_t1_name', type=str, default=None, help='image 1 name')
parser.add_argument('--data_t2_name', dest='data_t2_name', type=str, default=None, help='image 2 name')
parser.add_argument('--reference_t1_name', dest='reference_t1_name', type=str, default=None, help='reference 1 name')
parser.add_argument('--reference_t2_name', dest='reference_t2_name', type=str, default=None, help='reference 2 name')
#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')
parser.add_argument('--checkpoint_results_main_path', dest='checkpoint_results_main_path', type=str, default='E:/PEDROWORK/Trabajo_Domain_Adaptation/Code/checkpoints_results/')

parser.add_argument('--discriminate_domain_targets', dest='discriminate_domain_targets', type=eval, choices=[True, False], default=False, help='Applies for Multi-target training. Decides whether each target dataset will be assigned a different domain label or every target dataset will get the same label.')

parser.add_argument('--num_domains', dest='num_domains', type=int, default=None, help='Number of targets for discriminator training (domain adaptation)')

args = parser.parse_args()

def main():

    if args.phase == SharedParameters.PHASE_TEST:
        print(args)
        if not os.path.exists(args.checkpoint_results_main_path + 'results/'):
            os.makedirs(args.checkpoint_results_main_path + 'results/')

        args.results_dir = args.checkpoint_results_main_path + 'results/' + args.results_dir + '/'
        args.checkpoint_dir = args.checkpoint_results_main_path + 'checkpoints/' + args.checkpoint_dir + '/'

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

        print(f'Cleaning up {args.results_dir}')
        cleanup_folder(args.results_dir)

        dataset = []

        if args.dataset == AMAZON_RO.DATASET:            
            dataset.append(AMAZON_RO(args))
        if args.dataset == AMAZON_PA.DATASET:            
            dataset.append(AMAZON_PA(args))
        if args.dataset == CERRADO_MA.DATASET:
            dataset.append(CERRADO_MA(args))

        if len(dataset) == 0:
            raise Exception("Invalid dataset argument.")

        for ds in dataset:
            ds.Tiles_Configuration(args, 0)
            ds.Coordinates_Creator(args, 0)

        print(f"[*] Iterating over checkpoint_files on {args.checkpoint_dir}")        
        
        checkpoint_files = [item for item in os.listdir(args.checkpoint_dir) if os.path.isdir(os.path.join(args.checkpoint_dir,item))]

        for i in range(len(checkpoint_files)):

            print(checkpoint_files[i])

            model_folder = checkpoint_files[i]            

            args.trained_model_path = os.path.join(args.checkpoint_dir,model_folder)
            model_folder_fields = model_folder.split('_')

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
            args.save_results_dir = os.path.join(args.results_dir, args.classifier_type + '_' + 'Model_Results_' + 'Trained_' + model_folder_fields[3] + '_' + model_folder_fields[4] + '_' + model_folder[-19:] + '_Tested_' + args.data_t1_year + '_' + args.data_t2_year + '_' + dt_string)
            
            print(f'Testing checkpoint {model_folder} at {dt_string}')            
            
            if not os.path.exists(args.save_results_dir):
                os.makedirs(args.save_results_dir)

            print('[*]Initializing the model...')

            try:
                model = Models(args, None, dataset)
                model.Test()

                print(f"Results have been generated successfully at {args.save_results_dir}")
            except Exception as e:
                print(e)                

if __name__=='__main__':
    main()
