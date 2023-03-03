import os
import sys
import skimage
import numpy as np
import scipy.io as sio
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from contextlib import redirect_stdout
from da_model import DomainAdaptationModel, MainNetwork, DomainRegressorNetwork
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from loss_functions import WeightedCrossEntropyC
from keras.backend import clear_session
from SharedParameters import TRAINING_TYPE_DOMAIN_ADAPTATION,TRAINING_TYPE_CLASSIFICATION
from Tools import Data_Augmentation_Definition, Patch_Extraction, Data_Augmentation_Execution, Classification_Maps, compute_metrics, mask_creation
from Networks import *
from discriminators import *
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class Models():
    def __init__(self, args, dataset_s, dataset_t):
        self.args = args
        self.dataset_s = dataset_s
        self.dataset_t = dataset_t

        if self.args.discriminate_domain_targets:
            if self.args.num_targets is None:
                raise Exception("Num_targets must have an integer positive value.")

            self.num_targets = self.args.num_targets + 1
        else:
            self.num_targets = 2

        print("self.discriminate_domain_targets: " + str(self.args.discriminate_domain_targets))
        print("self.num_targets: " + str(self.num_targets))

        self.checkpoint_name = self.args.classifier_type        

        self.classifier_loss = WeightedCrossEntropyC()
        self.domainregressor_loss = CategoricalCrossentropy(from_logits=True) 
        self.acc_function_discriminator = CategoricalAccuracy()   
        self.acc_function_discriminator_val = CategoricalAccuracy()               

        self.segmentation_history = {}
        self.discriminator_history = {}

        self.segmentation_history["loss"] = []        
        self.segmentation_history["f1"] = []
        self.segmentation_history["val_loss"] = []
        self.segmentation_history["val_f1"] = []

        self.discriminator_history["loss"] = []
        self.discriminator_history["val_loss"] = []
        self.discriminator_history["accuracy"] = []
        self.discriminator_history["val_accuracy"] = []

        if args.compute_ndvi:
            self.input_shape = (int(args.patches_dimension), int(args.patches_dimension), 2 * int(args.image_channels) + 2)
        else:
            self.input_shape = (int(args.patches_dimension), int(args.patches_dimension), 2 * int(args.image_channels))

        if self.args.phase == 'train':
            self.model = self.assembly_empty_model(showSummary=True)
        
        elif self.args.phase == 'test':
            print('[*] Loading the feature extractor and classifier trained models...')            
            try:
                print("[*] Loading weights from {0}".format(self.args.trained_model_path))
                self.load_weights(self.args.trained_model_path)            
                print('[*] Weights loaded successfuly.')
            except Exception as e:
                raise Exception('[!] Load failed... Details: {0}'.format(e))                

    def assembly_empty_model(self, showSummary: bool = False):
        if self.args.classifier_type == 'DeepLab':
            self.args.backbone = 'xception'
            self.args.aspp_rates = (1, 2, 3)
            self.args.data_format = 'channel_last'
            self.args.bn_decay = 0.9997

            deepLab = DeepLabV3Plus(self.args)

            print("Assemblying model")
            print(f"Input shape: {self.input_shape}")

            input_block = tf.keras.layers.Input(shape = self.input_shape)

            #Building the encoder            
            Encoder_Outputs, low_Level_Features = deepLab.build_DeepLab_Encoder(input_block)
            encoder_model = tf.keras.Model(inputs = input_block, outputs = [Encoder_Outputs, low_Level_Features], name = 'deeplabv3plus_encoder')
 
            #Building Decoder
            Decoder_Outputs = deepLab.build_DeepLab_Decoder([Encoder_Outputs, low_Level_Features])            
            decoder_model = tf.keras.Model(inputs = [Encoder_Outputs,low_Level_Features], outputs = Decoder_Outputs, name = 'deeplabv3plus_decoder')

            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:
                if self.args.domain_regressor_type.casefold() == 'fc':
                    discriminator_model = Domain_Regressor_FullyConnected(input_shape=Encoder_Outputs.shape[1:],units=1024, num_targets=self.num_targets)                    
                if self.args.domain_regressor_type.casefold() == 'conv':
                    discriminator_model = Domain_Regressor_Convolutional(input_shape=Encoder_Outputs.shape[1:], num_targets=self.num_targets)

                self.D_out_shape = discriminator_model.layers[-2].output.shape[1:]

                print(f'D_out_shape: {self.D_out_shape}')

                empty_model = DomainAdaptationModel(self.input_shape, encoder_model,decoder_model, discriminator_model)                

                if showSummary:
                    self.summary(empty_model.encoder_model, "Encoder: ")
                    self.summary(empty_model.decoder_model, "Decoder: ")
                    self.summary(empty_model.domain_discriminator, "Domain_Regressor: ")

            else:
                empty_model = tf.keras.Model(inputs = input_block, outputs = Decoder_Outputs, name = 'deeplabv3plus')
                if showSummary:
                    self.summary(empty_model, "Encoder/Decoder: ")                      
        else:
            raise Exception("Invalid or not implemented classifier_type: {0}".format(self.args.classifier_type))        
        
        return empty_model

    @tf.function
    def _training_step(self, x_batch_train, y_batch_train, class_weights, mask_c):
        with tf.GradientTape() as tape:
            self.classifier_loss.class_weights = class_weights
            self.classifier_loss.mask = mask_c
            y_pred = self.model(x_batch_train,training = True)            
            loss = self.classifier_loss(y_batch_train, y_pred)
              
        self.training_optimizer.minimize(loss, self.model.trainable_weights,tape=tape)        
        return loss, y_pred

    @tf.function
    def _test_step(self, x_batch_train, y_batch_train, class_weights, mask_c):
        self.classifier_loss.class_weights = class_weights
        self.classifier_loss.mask = mask_c
        y_pred = self.model(x_batch_train, training = False)
        loss = self.classifier_loss(y_batch_train, y_pred)
        return loss, y_pred

    @tf.function
    def _training_step_domain_adaptation(self, inputs, outputs, class_weights, mask_c):

        y_true_segmentation, y_true_discriminator = outputs
        x_input, l = inputs
        
        self.classifier_loss.class_weights = class_weights
        self.classifier_loss.mask = mask_c        
        
        with tf.GradientTape(persistent=True) as tape:
            y_pred_segmentation,_,logits_discriminator = self.model([x_input, l],training=True)
            
            loss_segmentation = self.classifier_loss(y_true_segmentation, y_pred_segmentation)            
            loss_discriminator = self.domainregressor_loss(y_true=y_true_discriminator, y_pred=logits_discriminator)            
            
            total_loss = loss_segmentation + loss_discriminator
    
        self.training_optimizer.minimize(total_loss, self.model.trainable_weights,tape=tape)
        
        self.acc_function_discriminator.update_state(y_true_discriminator, logits_discriminator)

        return loss_segmentation, y_pred_segmentation, loss_discriminator
		
		
    @tf.function
    def _test_step_domain_adaptation(self, inputs, outputs, class_weights, mask_c):
        x_input, _ = inputs
        y_true_segmentation, y_true_discriminator = outputs       
    
        self.classifier_loss.class_weights = class_weights
        self.classifier_loss.mask = mask_c

        y_pred_segmentation,_,logits_discriminator = self.model(inputs,training=False)
        
        loss_segmentation = self.classifier_loss(y_true_segmentation, y_pred_segmentation)
        loss_discriminator = self.domainregressor_loss(y_true=y_true_discriminator, y_pred=logits_discriminator)             

        self.acc_function_discriminator_val.update_state(y_true_discriminator, logits_discriminator)

        return loss_segmentation, y_pred_segmentation, loss_discriminator

    def summary(self, net, name):
        print(name)
        net.summary()
        summaryFile = os.path.join(self.args.save_checkpoint_path,"Architecture.txt")        
        with open(summaryFile, 'a') as f:
            f.write(name + "\n")
            with redirect_stdout(f):
                net.summary()

    def weighted_cross_entropy_c(self, label_c, prediction_c, class_weights):
        temp = -label_c * tf.log(prediction_c + 1e-3)
        temp_weighted = class_weights * temp
        loss = tf.reduce_sum(temp_weighted, 3)
        return loss

    def learning_rate_decay(self,lr_0):
        return lr_0 / (1. + 10 * self.p)**0.75

    def Train(self):

        best_val_fs = 0
        best_val_dr = 0
        best_mod_fs = 0
        best_mod_dr = 0
        #best_f1score = 0
        pat = 0
        
        class_weights = []
        class_weights.append(0.4)
        class_weights.append(2)

        reference_t1_s = np.zeros((self.dataset_s.references_[0].shape[0], self.dataset_s.references_[0].shape[1], 1))
        reference_t2_s = np.zeros((self.dataset_s.references_[0].shape[0], self.dataset_s.references_[0].shape[1], 1))
        
        reference_t1_t = []
        reference_t2_t = []

        for t in self.dataset_t:
            reference_t1_t.append(np.zeros((t.references_[0].shape[0], t.references_[0].shape[1], 1)))
            reference_t2_t.append(np.zeros((t.references_[0].shape[0], t.references_[0].shape[1], 1)))

        if self.args.balanced_tr:
            class_weights = self.dataset_s.class_weights

        # Copy the original input values
        corners_coordinates_tr_s = self.dataset_s.corners_coordinates_tr.copy()
        corners_coordinates_vl_s = self.dataset_s.corners_coordinates_vl.copy()
        reference_t1_ = self.dataset_s.references_[0].copy()
        reference_t1_[self.dataset_s.references_[0] == 0] = 1
        reference_t1_[self.dataset_s.references_[0] == 1] = 0

        reference_t1_s[:,:,0] = reference_t1_.copy()
        reference_t2_s[:,:,0] = self.dataset_s.references_[1].copy()

        corners_coordinates_tr_t = []
        corners_coordinates_vl_t = []

        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:

            for t in self.dataset_t:
                corners_coordinates_tr_t.append(t.corners_coordinates_tr.copy())
                corners_coordinates_vl_t.append(t.corners_coordinates_vl.copy())

            if 'CL' in self.args.da_type:
                print("CL mode domain adaptation - Target domain will provide labels for training")
                for i in range(len(self.dataset_t)):
                    reference_t1_ = self.dataset_t[i].references_[0].copy()
                    reference_t1_[self.dataset_t[i].references_[0] == 0] = 1
                    reference_t1_[self.dataset_t[i].references_[0] == 1] = 0
                    reference_t1_t[i][:,:,0] = reference_t1_.copy()
                    reference_t2_t[i][:,:,0] = self.dataset_t[i].references_[1].copy()


        print('Sets dimensions before data augmentation')
        print('Source dimensions: ')
        print(np.shape(corners_coordinates_tr_s))
        print(np.shape(corners_coordinates_vl_s))
        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            print('Target dimension: ')
            print(np.shape(corners_coordinates_tr_t))
            print(np.shape(corners_coordinates_tr_t[0]))
            print(np.shape(corners_coordinates_vl_t))
            print(np.shape(corners_coordinates_vl_t[0]))
            print('******************************')
        
        if self.args.data_augmentation:
            corners_coordinates_tr_s = Data_Augmentation_Definition(corners_coordinates_tr_s)
            corners_coordinates_vl_s = Data_Augmentation_Definition(corners_coordinates_vl_s)
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                for i in range(len(self.dataset_t)):
                    corners_coordinates_tr_t[i] = Data_Augmentation_Definition(corners_coordinates_tr_t[i])
                    corners_coordinates_vl_t[i] = Data_Augmentation_Definition(corners_coordinates_vl_t[i])
                
        if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:
            #Generating target labels before data shuffling and balancing            
            # Target Domain labels configuration
            # Source: 0
            # Target 1: 1
            # Target 2: 2                
            target_labels_tr = []                
            target_labels_vl = []     

            domain_indexs_tr_t = []
            domain_indexs_vl_t = []

            #size_tr_t = []
            #size_vl_t = []
            
            target_label_value = 1                    
            for i in range(len(self.dataset_t)): 
                print(f"Target Dataset {self.dataset_t[i].DATASET}")

                if self.args.discriminate_domain_targets:
                    print("Assigned label value: %d"%(target_label_value))
                    if len(self.D_out_shape) > 2:
                        target_labels_tr.append(np.full((corners_coordinates_tr_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1),target_label_value))
                        target_labels_vl.append(np.full((corners_coordinates_vl_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1),target_label_value))
                    else:
                        target_labels_tr.append(np.full((corners_coordinates_tr_t[i].shape[0], 1), target_label_value))
                        target_labels_vl.append(np.full((corners_coordinates_vl_t[i].shape[0], 1), target_label_value))
                else:
                    print("Assigned label value: 1")
                    if len(self.D_out_shape) > 2:
                        target_labels_tr.append(np.ones((corners_coordinates_tr_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                        target_labels_vl.append(np.ones((corners_coordinates_vl_t[i].shape[0], self.D_out_shape[0], self.D_out_shape[1],1)))
                    else:
                        target_labels_tr.append(np.ones((corners_coordinates_tr_t[i].shape[0], 1)))
                        target_labels_vl.append(np.ones((corners_coordinates_vl_t[i].shape[0], 1)))
                
                # Target Domains indexes configuration                    
                domain_indexs_tr_t.append(np.full((corners_coordinates_tr_t[i].shape[0], 1), target_label_value))
                domain_indexs_vl_t.append(np.full((corners_coordinates_vl_t[i].shape[0], 1), target_label_value))
                
                
                target_label_value += 1

            #if self.args.source_targets_balanced == True:
            size_tr_s = [np.shape(corners_coordinates_tr_s)[0]]
            size_vl_s = [np.shape(corners_coordinates_vl_s)[0]]

            size_tr_t = [x.shape[0] for x in corners_coordinates_tr_t]
            size_vl_t = [x.shape[0] for x in corners_coordinates_vl_t]

            size_tr_list = np.concatenate((size_tr_s,size_tr_t),axis=0)
            size_vl_list = np.concatenate((size_vl_s,size_vl_t),axis=0)

            index_min_size_tr = np.argmin(size_tr_list)
            index_min_size_vl = np.argmin(size_vl_list)

            min_tr_size = size_tr_list[index_min_size_tr]
            min_vl_size = size_vl_list[index_min_size_vl]  

            print('size_tr_list:')
            print(size_tr_list)

            print('size_vl_list:')
            print(size_vl_list)


            #Combines different targets into one dataset
            #corners_coordinates_tr_t = np.concatenate(corners_coordinates_tr_t, axis=0)
            #corners_coordinates_vl_t = np.concatenate(corners_coordinates_vl_t, axis=0)

            # Balancing the number of samples between source and target domain
            #size_tr_s = np.shape(corners_coordinates_tr_s)[0]
            #size_tr_t = np.shape(corners_coordinates_tr_t)[0]
            #size_vl_s = np.shape(corners_coordinates_vl_s)[0]
            #size_vl_t = np.shape(corners_coordinates_vl_t)[0]            

            #Shuffling the num_samples
            index_tr_s = np.arange(size_tr_s[0])
            index_vl_s = np.arange(size_vl_s[0])

            np.random.shuffle(index_tr_s)            
            np.random.shuffle(index_vl_s)            

            corners_coordinates_tr_s = corners_coordinates_tr_s[index_tr_s, :]
            corners_coordinates_vl_s = corners_coordinates_vl_s[index_vl_s, :]    

            #RETIRA AMOSTRAS DO CONJUNTO MAIOR PARA SE IGUALAR AO MENOR            
            corners_coordinates_tr_s = corners_coordinates_tr_s[:min_tr_size,:]    
            corners_coordinates_vl_s = corners_coordinates_vl_s[:min_vl_size,:]        

            index_tr_t = []
            index_vl_t = []
            
            for i in range(len(self.dataset_t)):
                index_tr_t.append(np.arange(size_tr_t[i]))
                index_vl_t.append(np.arange(size_vl_t[i]))
                
                np.random.shuffle(index_tr_t[i])
                np.random.shuffle(index_vl_t[i])

                corners_coordinates_tr_t[i] = corners_coordinates_tr_t[i][index_tr_t[i], :]            
                corners_coordinates_vl_t[i] = corners_coordinates_vl_t[i][index_vl_t[i], :]
            
                target_labels_tr[i] = target_labels_tr[i][index_tr_t[i], :]
                target_labels_vl[i] = target_labels_vl[i][index_vl_t[i], :]

                domain_indexs_tr_t[i] = domain_indexs_tr_t[i][index_tr_t[i], :]
                domain_indexs_vl_t[i] = domain_indexs_vl_t[i][index_vl_t[i], :]

                #RETIRA AMOSTRAS DO CONJUNTO MAIOR PARA SE IGUALAR AO MENOR      
                corners_coordinates_tr_t[i] = corners_coordinates_tr_t[i][:min_tr_size,:]
                target_labels_tr[i] = target_labels_tr[i][:min_tr_size, :]
                domain_indexs_tr_t[i] = domain_indexs_tr_t[i][:min_tr_size, :]
                    
                corners_coordinates_vl_t[i] = corners_coordinates_vl_t[i][:min_vl_size,:]
                target_labels_vl[i] = target_labels_vl[i][:min_vl_size,:]
                domain_indexs_vl_t[i] = domain_indexs_vl_t[i][:min_vl_size,:]             

        data = []
        x_train_s = np.concatenate((self.dataset_s.images_norm_[0], self.dataset_s.images_norm_[1], reference_t1_s, reference_t2_s), axis = 2)
        data.append(x_train_s)

        # Training configuration
        if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
            # Domain indexs configuration
            corners_coordinates_tr = corners_coordinates_tr_s.copy()
            corners_coordinates_vl = corners_coordinates_vl_s.copy()
            domain_indexs_tr = np.zeros((np.shape(corners_coordinates_tr)[0], 1))
            domain_indexs_vl = np.zeros((np.shape(corners_coordinates_vl)[0], 1))

        elif self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:            
            for i in range(len(self.dataset_t)):
                x_train_t = np.concatenate((self.dataset_t[i].images_norm_[0], self.dataset_t[i].images_norm_[1], reference_t1_t[i], reference_t2_t[i]), axis = 2)
                data.append(x_train_t)
                       
            corners_coordinates_tr_t = np.concatenate(corners_coordinates_tr_t, axis=0)
            corners_coordinates_vl_t = np.concatenate(corners_coordinates_vl_t, axis=0)

            corners_coordinates_tr = np.concatenate((corners_coordinates_tr_s, corners_coordinates_tr_t), axis = 0)
            corners_coordinates_vl = np.concatenate((corners_coordinates_vl_s, corners_coordinates_vl_t), axis = 0)
            
            # Source Domain indexs configuration
            domain_indexs_tr_s = np.zeros((np.shape(corners_coordinates_tr_s)[0], 1))            
            domain_indexs_vl_s = np.zeros((np.shape(corners_coordinates_vl_s)[0], 1))     

            domain_indexs_tr_t = np.concatenate(domain_indexs_tr_t,axis=0)
            domain_indexs_vl_t = np.concatenate(domain_indexs_vl_t,axis=0)        

            domain_indexs_tr = np.concatenate((domain_indexs_tr_s, domain_indexs_tr_t), axis = 0)
            domain_indexs_vl = np.concatenate((domain_indexs_vl_s, domain_indexs_vl_t), axis = 0)

            target_labels_tr = np.concatenate(target_labels_tr,axis=0)
            target_labels_vl = np.concatenate(target_labels_vl,axis=0)


            if 'DR' in self.args.da_type:
                # Source Domain labels configuration
                if len(self.D_out_shape) > 2:
                    source_labels_tr = np.zeros((np.shape(corners_coordinates_tr_s)[0], self.D_out_shape[0], self.D_out_shape[1],1))
                    source_labels_vl = np.zeros((np.shape(corners_coordinates_vl_s)[0], self.D_out_shape[0], self.D_out_shape[1],1))
                else:
                    source_labels_tr = np.zeros((corners_coordinates_tr_s.shape[0], 1))
                    source_labels_vl = np.zeros((corners_coordinates_vl_s.shape[0], 1))
                
                print(f"Source Dataset {self.dataset_s.DATASET}")
                print("Assigned label value: 0")
                
                y_train_d = np.concatenate((source_labels_tr, target_labels_tr), axis = 0)
                y_valid_d = np.concatenate((source_labels_vl, target_labels_vl), axis = 0)

        #Computing the number of batches
        num_batches_tr = corners_coordinates_tr.shape[0]//self.args.batch_size
        num_batches_vl = corners_coordinates_vl.shape[0]//self.args.batch_size        

        self.training_optimizer = Adam(learning_rate=MyDecay(num_batches_tr, self.args.epochs, self.args.lr), beta_1=self.args.beta1)

        #Training starts now:
        e = 0
        best_model_epoch = -1
        while (e < self.args.epochs):     
            self.acc_function_discriminator.reset_states()
            self.acc_function_discriminator_val.reset_states()

            #Shuffling the data and the labels
            num_samples = corners_coordinates_tr.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            corners_coordinates_tr = corners_coordinates_tr[index, :]
            domain_indexs_tr = domain_indexs_tr[index, :]
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                if 'DR' in self.args.da_type:
                    if len(self.D_out_shape) > 2:
                        y_train_d = y_train_d[index, :, :, :]
                    else:
                        y_train_d = y_train_d[index, :]

            #Shuffling the data and the labels for validation samples
            num_samples = corners_coordinates_vl.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            corners_coordinates_vl = corners_coordinates_vl[index, :]
            domain_indexs_vl = domain_indexs_vl[index, :]
            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                if 'DR' in self.args.da_type:
                    if len(self.D_out_shape) > 2:
                        y_valid_d = y_valid_d[index, :, :, :]
                    else:
                        y_valid_d = y_valid_d[index, :]

            # Open a file in order to save the training history
            with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                #Initializing loss metrics
                loss_cl_tr = np.zeros((1 , 2))
                loss_cl_vl = np.zeros((1 , 2))
                loss_dr_tr = np.zeros((1 , 2))
                loss_dr_vl = np.zeros((1 , 2))

                accuracy_tr = 0
                f1_score_tr = 0
                recall_tr = 0
                precission_tr = 0

                accuracy_vl = 0
                f1_score_vl = 0
                recall_vl = 0
                precission_vl = 0

                print("----------------------------------------------------------")
                #Computing some parameters
                self.p = float(e) / self.args.epochs
                print("Current Epoch: {0}/{1}".format(str(e),str(self.args.epochs)))

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                    warmup = self.args.warmup
                    print("Number of warm-up epochs: %d"%(warmup))
                    if e >= warmup:
                        self.l = 2. / (1. + np.exp(-2.5 * self.p)) - 1                        
                        #self.l = 2. / (1. + np.exp(-10 * self.p)) - 1
                    else:
                        self.l = 0.
                    print("lambda_p: %.6f" %(self.l))

                lr = self.learning_rate_decay(self.args.lr)
                
                print("Learning rate decay: %.6f"%(lr))                

                batch_counter_cl = 0
                batchs = trange(num_batches_tr)
                #for b in range(num_batches_tr):
                for b in batchs:
                    corners_coordinates_tr_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                    domain_index_batch = domain_indexs_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                    if self.args.data_augmentation:
                        transformation_indexs_batch = corners_coordinates_tr[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]

                    #Extracting the data patches from it's coordinates
                    data_batch_ = Patch_Extraction(data, corners_coordinates_tr_batch, domain_index_batch, self.args.patches_dimension)

                    # Perform data augmentation?
                    if self.args.data_augmentation:
                        data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)
                    # Recovering data
                    data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                    # Recovering past reference
                    reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                    reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]
                    # plt.imshow(reference_t1_[0,:,:])
                    # plt.show()
                    # plt.imshow(reference_t2_[0,:,:])
                    # plt.show()
                    # Hot encoding the reference_t2_
                    y_train_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                    classification_mask_batch = reference_t1_.copy()

                    # Setting the class weights
                    Weights = np.ones((self.args.batch_size, self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                    Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]

                    if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
                        c_batch_loss, batch_probs = self._training_step(data_batch, y_train_c_hot_batch, Weights, classification_mask_batch)
                    elif self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                        if 'DR' in self.args.da_type:
                            if len(self.D_out_shape) > 2:
                                y_train_d_batch = y_train_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :, :,:]
                            else:
                                y_train_d_batch = y_train_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                            y_train_d_hot_batch = tf.keras.utils.to_categorical(y_train_d_batch, self.num_targets) 

                            c_batch_loss, batch_probs, d_batch_loss = self._training_step_domain_adaptation([data_batch,self.l], [y_train_c_hot_batch, y_train_d_hot_batch], Weights, classification_mask_batch)

                            loss_dr_tr[0 , 0] += d_batch_loss

                            acc_discriminator_train = float(self.acc_function_discriminator.result())
                        else:
                            c_batch_loss, batch_probs = self._training_step(data_batch, y_train_c_hot_batch, Weights, classification_mask_batch)
                    
                    loss_cl_tr[0 , 0] += c_batch_loss
                    # print(loss_cl_tr)
                    y_train_predict_batch = np.argmax(batch_probs, axis = 3)
                    y_train_batch = np.argmax(y_train_c_hot_batch, axis = 3)

                    # Reshaping probability output, true labels and last reference
                    y_train_predict_r = y_train_predict_batch.reshape((y_train_predict_batch.shape[0] * y_train_predict_batch.shape[1] * y_train_predict_batch.shape[2], 1))
                    y_train_true_r = y_train_batch.reshape((y_train_batch.shape[0] * y_train_batch.shape[1] * y_train_batch.shape[2], 1))
                    classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))

                    available_training_pixels= np.transpose(np.array(np.where(classification_mask_batch_r == 1)))

                    y_predict = y_train_predict_r[available_training_pixels[:,0],available_training_pixels[:,1]]
                    y_true = y_train_true_r[available_training_pixels[:,0],available_training_pixels[:,1]]

                    accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))

                    accuracy_tr += accuracy
                    f1_score_tr += f1score
                    recall_tr += recall
                    precission_tr += precission

                    batch_counter_cl += 1

                loss_cl_tr = loss_cl_tr/batch_counter_cl
                accuracy_tr = accuracy_tr/batch_counter_cl
                f1_score_tr = f1_score_tr/batch_counter_cl
                recall_tr = recall_tr/batch_counter_cl
                precission_tr = precission_tr/batch_counter_cl

                self.segmentation_history["loss"].append(loss_cl_tr[0,0])       
                self.segmentation_history["f1"].append(f1_score_tr)

                print("batch_counter_cl:")
                print(batch_counter_cl)

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:                    
                    loss_dr_tr = loss_dr_tr/batch_counter_cl
                    self.discriminator_history["loss"].append(loss_dr_tr[0,0])
                    self.discriminator_history["accuracy"].append(acc_discriminator_train)

                    print ("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, Dr loss: %f, Dr accuracy: %.2f]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0], acc_discriminator_train))
                    f.write("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, Dr loss: %f, Dr accuracy: %.2f]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr, loss_dr_tr[0,0], acc_discriminator_train))
                else:
                    print ("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))
                    f.write("%d [Training loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]\n" % (e, loss_cl_tr[0,0], accuracy_tr, precission_tr, recall_tr, f1_score_tr))

                #Computing the validation loss
                print('[*]Computing the validation loss...')
                batch_counter_cl = 0
                batchs = trange(num_batches_vl)
                #for b in range(num_batches_vl):
                for b in batchs:
                    corners_coordinates_vl_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                    domain_index_batch = domain_indexs_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]

                    if self.args.data_augmentation:
                        transformation_indexs_batch = corners_coordinates_vl[b * self.args.batch_size : (b + 1) * self.args.batch_size , 4]

                    #Extracting the data patches from it's coordinates
                    data_batch_ = Patch_Extraction(data, corners_coordinates_vl_batch, domain_index_batch, self.args.patches_dimension)

                    if self.args.data_augmentation:
                        data_batch_ = Data_Augmentation_Execution(data_batch_, transformation_indexs_batch)

                    # Recovering data
                    data_batch = data_batch_[:,:,:,: 2 * self.args.image_channels]
                    # Recovering past reference
                    reference_t1_ = data_batch_[:,:,:, 2 * self.args.image_channels]
                    reference_t2_ = data_batch_[:,:,:, 2 * self.args.image_channels + 1]

                    # Hot encoding the reference_t2_
                    y_valid_c_hot_batch = tf.keras.utils.to_categorical(reference_t2_, self.args.num_classes)
                    classification_mask_batch = reference_t1_.copy()

                    # Setting the class weights
                    Weights = np.ones((corners_coordinates_vl_batch.shape[0], self.args.patches_dimension, self.args.patches_dimension, self.args.num_classes),dtype=np.float32)
                    Weights[:,:,:,0] = class_weights[0] * Weights[:,:,:,0]
                    Weights[:,:,:,1] = class_weights[1] * Weights[:,:,:,1]
                    
                    if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
                        c_batch_loss, batch_probs = self._test_step(data_batch, y_valid_c_hot_batch, Weights, classification_mask_batch)
                    if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                        if 'DR' in self.args.da_type:
                            if len(self.D_out_shape) > 2:
                                y_valid_d_batch = y_valid_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :, :,:]
                            else:
                                y_valid_d_batch = y_valid_d[b * self.args.batch_size : (b + 1) * self.args.batch_size, :]
                            
                            y_valid_d_hot_batch = tf.keras.utils.to_categorical(y_valid_d_batch, self.num_targets)

                            c_batch_loss, batch_probs, d_batch_loss = self._test_step_domain_adaptation([data_batch,self.l], [y_valid_c_hot_batch, y_valid_d_hot_batch], Weights, classification_mask_batch)

                            loss_dr_vl[0 , 0] += d_batch_loss

                            acc_discriminator_val = float(self.acc_function_discriminator_val.result())
                        else:
                            c_batch_loss, batch_probs = self._test_step(data_batch, y_valid_c_hot_batch, Weights, classification_mask_batch)                            

                    loss_cl_vl[0 , 0] += c_batch_loss

                    y_valid_batch = np.argmax(y_valid_c_hot_batch, axis = 3)
                    y_valid_predict_batch = np.argmax(batch_probs, axis = 3)

                    # Reshaping probability output, true labels and last reference
                    y_valid_predict_r = y_valid_predict_batch.reshape((y_valid_predict_batch.shape[0] * y_valid_predict_batch.shape[1] * y_valid_predict_batch.shape[2], 1))
                    y_valid_true_r = y_valid_batch.reshape((y_valid_batch.shape[0] * y_valid_batch.shape[1] * y_valid_batch.shape[2], 1))
                    classification_mask_batch_r = classification_mask_batch.reshape((classification_mask_batch.shape[0] * classification_mask_batch.shape[1] * classification_mask_batch.shape[2], 1))

                    available_validation_pixels = np.transpose(np.array(np.where(classification_mask_batch_r == 1)))

                    y_predict = y_valid_predict_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]
                    y_true = y_valid_true_r[available_validation_pixels[:,0],available_validation_pixels[:,1]]

                    accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_true.astype(int), y_predict.astype(int))

                    accuracy_vl += accuracy
                    f1_score_vl += f1score
                    recall_vl += recall
                    precission_vl += precission
                    batch_counter_cl += 1

                loss_cl_vl = loss_cl_vl/(batch_counter_cl)
                accuracy_vl = accuracy_vl/(batch_counter_cl)
                f1_score_vl = f1_score_vl/(batch_counter_cl)
                recall_vl = recall_vl/(batch_counter_cl)
                precission_vl = precission_vl/(batch_counter_cl)

                self.segmentation_history["val_loss"].append(loss_cl_vl[0,0])      
                self.segmentation_history["val_f1"].append(f1_score_vl)

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION and 'DR' in self.args.da_type:                    
                    loss_dr_vl = loss_dr_vl/batch_counter_cl
                    self.discriminator_history["val_loss"].append(loss_dr_vl[0,0])
                    self.discriminator_history["val_accuracy"].append(acc_discriminator_val)

                    print ("%d [Validation loss: %f, acc.: %.2f%%,  precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, DrV loss: %f, DrV accuracy: %.2f]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0 , 0], acc_discriminator_val))
                    f.write("%d [Validation loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%, DrV loss: %f, DrV accuracy: %.2f]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl, loss_dr_vl[0 , 0], acc_discriminator_val))
                else:
                    print ("%d [Validation loss: %f, acc.: %.2f%%,  precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))
                    f.write("%d [Validation loss: %f, acc.: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%]\n" % (e, loss_cl_vl[0,0], accuracy_vl, precission_vl, recall_vl, f1_score_vl))

            if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                    if 'DR' in self.args.da_type:
                        if np.isnan(loss_cl_tr[0,0]) or np.isnan(loss_cl_vl[0,0]):
                            print('Nan value detected!!!!')                            
                            if best_model_epoch != -1:
                                pat += 1
                                if pat > self.args.patience:
                                    print("Patience limit reached. Exiting training...")
                                    break
                                print('[*]ReLoading the models weights...')
                                try:                                   
                                    self.load_weights(self.args.save_checkpoint_path)                         
                                    print(" [*] Load successfuly")
                                except Exception as e:                                 
                                    print(" [!] Load failed... Details: {0}".format(e))
                        elif self.l != 0:
                            FLAG = False
                            if  best_val_dr < loss_dr_vl[0 , 0]:
                                if best_val_fs < f1_score_vl:
                                    best_val_dr = loss_dr_vl[0 , 0]
                                    best_val_fs = f1_score_vl
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best ideal model at epoch: ' + str(e))
                                    f.write("[!]Ideal best ideal model\n")                                    
                                    self.save_weights(self.args.save_checkpoint_path)
                                    FLAG = True
                                elif np.abs(best_val_fs - f1_score_vl) < 3:
                                    best_val_dr = loss_dr_vl[0 , 0]
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best model attending best Dr_loss at epoch: ' + str(e))
                                    f.write("[!]Best model attending best Dr_loss\n")                                    
                                    self.save_weights(self.args.save_checkpoint_path)
                                    FLAG = True
                            elif best_val_fs < f1_score_vl:
                                if  np.abs(best_val_dr - loss_dr_vl[0 , 0]) < 0.2:
                                    best_val_fs = f1_score_vl
                                    best_mod_fs = f1_score_vl
                                    best_mod_dr = loss_dr_vl[0 , 0]
                                    best_model_epoch = e
                                    print('[!]Saving best model attending best f1-score at epoch: ' + str(e))
                                    f.write("[!]Best model attending best f1-score \n")                                    
                                    self.save_weights(self.args.save_checkpoint_path)
                                    FLAG = True

                            if FLAG:
                                pat = 0
                                print('[!] Best Model with DrV loss: %.3f and F1-Score: %.2f%%'% (best_mod_dr, best_mod_fs))
                            else:
                                print('[!] The Model has not been considered as suitable for saving procedure.')
                                if best_model_epoch != -1:
                                    pat += 1
                                    if pat > self.args.patience:
                                        break
                        else:
                            print("Warming up!")
                    else:
                        if best_val_fs < f1_score_vl:
                            best_val_fs = f1_score_vl
                            pat = 0
                            print('[!] Best Validation F1 score: %.2f%%'%(best_val_fs))
                            best_model_epoch = e
                            if self.args.save_intermediate_model:
                                print('[!]Saving best model at epoch: ' + str(e))                                
                                self.save_weights(self.args.save_checkpoint_path)
                        else:
                            pat += 1
                            if pat > self.args.patience:
                                print("Patience limit reached. Exiting training...")
                                break
            else:
                if best_val_fs < f1_score_vl:
                    best_val_fs = f1_score_vl
                    pat = 0
                    print('[!] Best Validation F1 score: %.2f%%'%(best_val_fs))
                    best_model_epoch = e
                    if self.args.save_intermediate_model:
                        print('[!]Saving best model at epoch: ' + str(e))                        
                        self.save_weights(self.args.save_checkpoint_path)
                else:
                    pat += 1
                    if pat > self.args.patience:
                        print("Patience limit reachead. Exiting training...")
                        break             
            e += 1

        
        if self.args.training_type == TRAINING_TYPE_CLASSIFICATION:
            self.plot_metrics_segmentation()
            self.plot_f1score_segmentation()
            with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                print('Training ended')
                f.write("Training ended\n")
                print("[!] Best Validation F1 score: %.2f%%"%(best_val_fs))
                f.write("[!] Best Validation F1 score: %.2f%%"%(best_val_fs))

        elif self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
            self.plot_metrics_segmentation()
            self.plot_f1score_segmentation()            

            with open(os.path.join(self.args.save_checkpoint_path,"Log.txt"),"a") as f:
                if best_model_epoch != -1:
                    print("Training ended")
                    print("[!] Best epoch: %d" %(best_model_epoch))
                    print("[!] Domain Regressor Validation F1-score: %.2f%%" % (best_mod_fs))
                    print("[!] DrV loss: %.3f" % (best_val_dr))
                    print("[!] Best DR Validation for higher DrV loss: %.3f and F1-Score: %.2f%%" % (best_mod_dr, best_mod_fs))

                    f.write("Training ended\n")
                    f.write("[!] Best epoch: %d\n" %(best_model_epoch))
                    f.write("[!] Domain Regressor Validation F1-score: %.2f%%: \n" % (best_mod_fs))
                    f.write("[!] DrV loss: %.3f: \n" % (best_val_dr))
                else:
                    print("Training ended")
                    print("[!] [!] No Model has been selected.")
                    print("loss_dr_vl: %.3f"%(loss_dr_vl[0 , 0]))                    
                    print("f1_score_vl: %.3f"%(f1_score_vl))                    

                    f.write("Training ended")
                    f.write("[!] [!] No Model has been selected.")
                    f.write("loss_dr_vl: %.3f \n"%(loss_dr_vl[0 , 0]))                    
                    f.write("f1_score_vl: %.3f \n"%(f1_score_vl))
        

    def Test(self):

        for ds in self.dataset_t:

            hit_map_ = np.zeros((ds.k1 * ds.stride, ds.k2 * ds.stride))

            x_test = []
            data = np.concatenate((ds.images_norm_[0], ds.images_norm_[1]), axis = 2)
            x_test.append(data)

            num_batches_ts = ds.corners_coordinates_ts.shape[0]//self.args.batch_size
            batchs = trange(num_batches_ts)
            print(num_batches_ts)

            for b in batchs:
                self.corners_coordinates_ts_batch = ds.corners_coordinates_ts[b * self.args.batch_size : (b + 1) * self.args.batch_size , :]
                #self.x_test_batch = Patch_Extraction(x_test, self.central_pixels_coor_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension, True, 'reflect')
                self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.args.batch_size , 1)), self.args.patches_dimension)
                
                #probs = self.sess.run(self.prediction_c,feed_dict={self.data: self.x_test_batch})

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                    probs = self.model.main_network.predict(self.x_test_batch)
                else:
                    probs = self.model.predict(self.x_test_batch)

                for i in range(self.args.batch_size):
                    hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(ds.stride),
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(ds.stride)] = probs[i, int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),
                                                                                                                                                           int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),1]

            if (num_batches_ts * self.args.batch_size) < ds.corners_coordinates_ts.shape[0]:
                self.corners_coordinates_ts_batch = ds.corners_coordinates_ts[num_batches_ts * self.args.batch_size : , :]
                self.x_test_batch = Patch_Extraction(x_test, self.corners_coordinates_ts_batch, np.zeros((self.corners_coordinates_ts_batch.shape[0] , 1)), self.args.patches_dimension)

                #probs = self.sess.run(self.prediction_c,feed_dict={self.data: self.x_test_batch})

                if self.args.training_type == TRAINING_TYPE_DOMAIN_ADAPTATION:
                    probs = self.model.main_network.predict(self.x_test_batch)
                else:
                    probs = self.model.predict(self.x_test_batch)

                for i in range(self.corners_coordinates_ts_batch.shape[0]):
                    hit_map_[int(self.corners_coordinates_ts_batch[i, 0]) : int(self.corners_coordinates_ts_batch[i, 0]) + int(ds.stride),
                        int(self.corners_coordinates_ts_batch[i, 1]) : int(self.corners_coordinates_ts_batch[i, 1]) + int(ds.stride)] = probs[i, int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),
                                                                                                                                                           int(ds.overlap//2) : int(ds.overlap//2) + int(ds.stride),1]
            hit_map = hit_map_[:ds.k1 * ds.stride - ds.step_row, :ds.k2 * ds.stride - ds.step_col]
            
            print("Hit map:")
            print(np.shape(hit_map))
            np.save(os.path.join(self.args.save_results_dir,'hit_map'), hit_map)    

    def save_weights(self, weights_path: str):
        full_path = os.path.join(weights_path, self.checkpoint_name)        
        self.model.save_weights(full_path)
        print("Checkpoint saved successfuly!")

    def load_weights(self, weights_path: str, piece: str = None):
        print("[*] Reading checkpoint...")          
        self.model = self.assembly_empty_model()
        self.model.load_weights(os.path.join(weights_path,self.checkpoint_name))          

    def plot_metrics_segmentation(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.segmentation_history["loss"], label="decoder training loss")        
        plt.plot(self.segmentation_history["val_loss"], label="decoder validation loss")  
        plt.plot(self.discriminator_history["loss"], label="discriminator training loss")    
        plt.plot(self.discriminator_history["val_loss"], label="discriminator validation loss")   
        plt.plot(self.discriminator_history["accuracy"], label="discriminator training accuracy")    
        plt.plot(self.discriminator_history["val_accuracy"], label="discriminator validation accuracy")        
        plt.title("Loss/Accuracy evolution")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(self.args.save_checkpoint_path,"segmentation_metrics.png"))

    def plot_f1score_segmentation(self):
        plt.figure(figsize=(10, 10))        
        plt.plot(self.segmentation_history["f1"], label="training f1score")        
        plt.plot(self.segmentation_history["val_f1"], label="validation f1score")
        plt.plot(self.discriminator_history["accuracy"], label="discriminator training accuracy")    
        plt.plot(self.discriminator_history["val_accuracy"], label="discriminator validation accuracy") 

        plt.title("Segmentation F1 vs Discriminator Acc evolution")
        plt.xlabel("Epoch #")
        plt.ylabel("F1")
        plt.ylim([0, 100])
        plt.legend()
        plt.savefig(os.path.join(self.args.save_checkpoint_path,"segmentation_f1score.png"))
    

def Metrics_For_Test(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     Thresholds,
                     args):

    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1

    Probs_init = hit_map
    positive_map_init = np.zeros_like(Probs_init)

    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    CLASSIFICATION_MAPS = np.zeros((len(Thresholds), hit_map.shape[0], hit_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))


    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)


        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1

        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy

        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        #print(np.shape(central_pixels_coordinates_ts))
        #print(np.shape(central_pixels_coordinates_ts_))
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

        #print(np.shape(central_pixels_coordinates_ts))
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0

        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        if args.create_classification_map:
            Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)
            #plt.imshow(Classification_map)
            plt.savefig(os.path.join(save_path,'Classification_map.jpg'))

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP

        denominator = TN + FN + FP + TP

        Alert_area = 100*(numerator/denominator)
        print(f1score)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area

    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(os.path.join(save_path,'Accuracy'), ACCURACY)
        np.save(os.path.join(save_path,'Fscore'), FSCORE)
        np.save(os.path.join(save_path,'Recall'), RECALL)
        np.save(os.path.join(save_path,'Precission'), PRECISSION)
        np.save(os.path.join(save_path,'Confusion_matrix'), CONFUSION_MATRIX)
        np.save(os.path.join(save_path,'Alert_area'), ALERT_AREA)

    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)

    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA

def Metrics_For_Test_M(hit_map,
                     reference_t1, reference_t2,
                     Train_tiles, Valid_tiles, Undesired_tiles,
                     args):



    save_path = args.results_dir + args.file + '/'
    print('[*]Defining the initial central patches coordinates...')
    mask_init = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, [], [], [])
    mask_final = mask_creation(reference_t1.shape[0], reference_t1.shape[1], args.horizontal_blocks, args.vertical_blocks, Train_tiles, Valid_tiles, Undesired_tiles)

    #mask_final = mask_final_.copy()
    mask_final[mask_final == 1] = 0
    mask_final[mask_final == 3] = 0
    mask_final[mask_final == 2] = 1

    sio.savemat(os.path.join(save_path,'hit_map.mat') , {'hit_map': hit_map})
    Probs_init = hit_map
    positive_map_init = np.zeros_like(Probs_init)

    reference_t1_copy_ = reference_t1.copy()
    reference_t1_copy_ = reference_t1_copy_ - 1
    reference_t1_copy_[reference_t1_copy_ == -1] = 1
    reference_t1_copy_[reference_t2 == 2] = 0
    mask_f_ = mask_final * reference_t1_copy_
    sio.savemat(os.path.join(save_path,'mask_f_.mat') , {'mask_f_': mask_f_})
    sio.savemat(os.path.join(save_path,'reference_t2.mat') , {'reference': reference_t2})
    # Raul Implementation
    min_array = np.zeros((1 , ))
    Pmax = np.max(Probs_init[mask_f_ == 1])
    probs_list = np.arange(Pmax, 0, -Pmax/(args.Npoints - 1))
    Thresholds = np.concatenate((probs_list , min_array))

    print('Max probability value:')
    print(Pmax)
    print('Thresholds:')
    print(Thresholds)
    # Metrics containers
    ACCURACY = np.zeros((1, len(Thresholds)))
    FSCORE = np.zeros((1, len(Thresholds)))
    RECALL = np.zeros((1, len(Thresholds)))
    PRECISSION = np.zeros((1, len(Thresholds)))
    CONFUSION_MATRIX = np.zeros((2 , 2, len(Thresholds)))
    #CLASSIFICATION_MAPS = np.zeros((len(Thresholds), hit_map.shape[0], hit_map.shape[1], 3))
    ALERT_AREA = np.zeros((1 , len(Thresholds)))


    print('[*]The metrics computation has started...')
    #Computing the metrics for each defined threshold
    for th in range(len(Thresholds)):
        print(Thresholds[th])

        positive_map_init = np.zeros_like(hit_map)
        reference_t1_copy = reference_t1.copy()

        threshold = Thresholds[th]
        positive_coordinates = np.transpose(np.array(np.where(Probs_init >= threshold)))
        positive_map_init[positive_coordinates[:,0].astype('int'), positive_coordinates[:,1].astype('int')] = 1

        if args.eliminate_regions:
            positive_map_init_ = skimage.morphology.area_opening(positive_map_init.astype('int'),area_threshold = args.area_avoided, connectivity=1)
            eliminated_samples = positive_map_init - positive_map_init_
        else:
            eliminated_samples = np.zeros_like(hit_map)


        reference_t1_copy = reference_t1_copy + eliminated_samples
        reference_t1_copy[reference_t1_copy == 2] = 1
        reference_t1_copy = reference_t1_copy - 1
        reference_t1_copy[reference_t1_copy == -1] = 1
        reference_t1_copy[reference_t2 == 2] = 0
        mask_f = mask_final * reference_t1_copy

        #central_pixels_coordinates_ts, y_test = Central_Pixel_Definition_For_Test(mask_final_, reference_t1_copy, reference_t2, args.patches_dimension, 1, 'metrics')
        central_pixels_coordinates_ts_ = np.transpose(np.array(np.where(mask_f == 1)))
        #print(np.shape(central_pixels_coordinates_ts))
        #print(np.shape(central_pixels_coordinates_ts_))
        y_test = reference_t2[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]

        #print(np.shape(central_pixels_coordinates_ts))
        Probs = hit_map[central_pixels_coordinates_ts_[:,0].astype('int'), central_pixels_coordinates_ts_[:,1].astype('int')]
        Probs[Probs >= Thresholds[th]] = 1
        Probs[Probs <  Thresholds[th]] = 0

        accuracy, f1score, recall, precission, conf_mat = compute_metrics(y_test.astype('int'), Probs.astype('int'))

        #Classification_map, _, _ = Classification_Maps(Probs, y_test, central_pixels_coordinates_ts_, hit_map)

        TP = conf_mat[1 , 1]
        FP = conf_mat[0 , 1]
        TN = conf_mat[0 , 0]
        FN = conf_mat[1 , 0]
        numerator = TP + FP

        denominator = TN + FN + FP + TP

        Alert_area = 100*(numerator/denominator)
        #print(f1score)
        print(precission)
        print(recall)
        ACCURACY[0 , th] = accuracy
        FSCORE[0 , th] = f1score
        RECALL[0 , th] = recall
        PRECISSION[0 , th] = precission
        CONFUSION_MATRIX[: , : , th] = conf_mat
        #CLASSIFICATION_MAPS[th, :, :, :] = Classification_map
        ALERT_AREA[0 , th] = Alert_area

    #Saving the metrics as npy array
    if not args.save_result_text:
        np.save(os.path.join(save_path,'Accuracy'), ACCURACY)
        np.save(os.path.join(save_path,'Fscore'), FSCORE)
        np.save(os.path.join(save_path,'Recall'), RECALL)
        np.save(os.path.join(save_path,'Precission'), PRECISSION)
        np.save(os.path.join(save_path,'Confusion_matrix'), CONFUSION_MATRIX)
        #np.save(save_path + 'Classification_maps', CLASSIFICATION_MAPS)
        np.save(os.path.join(save_path,'Alert_area'), ALERT_AREA)

    print('Accuracy')
    print(ACCURACY)
    print('Fscore')
    print(FSCORE)
    print('Recall')
    print(RECALL)
    print('Precision')
    print(PRECISSION)
    print('Confusion matrix')
    print(CONFUSION_MATRIX[:,:,0])
    print('Alert_area')
    print(ALERT_AREA)

    return ACCURACY, FSCORE, RECALL, PRECISSION, CONFUSION_MATRIX, ALERT_AREA


class MyDecay(LearningRateSchedule):

    def __init__(self, batch_size, max_steps=100., mu_0=0.01, alpha=10, beta=0.75):
        self.batch_size = batch_size
        self.max_steps = max_steps        
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta        

    def __call__(self, step):        
        step = tf.cast(step, dtype=tf.float32)
        _step = step // self.batch_size
        p = _step / self.max_steps
        return self.mu_0 / (1 + self.alpha * p)**self.beta