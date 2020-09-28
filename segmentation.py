
import sys, os
from time import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

# Scikit stuff
from scipy.ndimage.measurements import label as connected_components
import sklearn
import sklearn.linear_model

# my libraries
import util.util as util_
import networks as networks
import evaluation
import hough_voting.hough_voting as hv
from pytorch_memlab import profile

# TQDM stuff
# from tqdm import tqdm_notebook as tqdm # Because i'm using jupyter notebook. This can be something else for command line usage
from tqdm import tqdm

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2

dont_send_to_device = ['scene_dir', 'view_num', 'subset', 'supporting_plane', 'label_abs_path']

class CombinedDSN(nn.Module):

    def __init__(self, encoder, decoder, fg_module, cd_module):
        super(CombinedDSN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fg_module = fg_module
        self.cd_module = cd_module

    def forward(self, xyz_img):
        """ Forward pass using entire DSN

            @param xyz_img: a [N x 3 x H x W] torch.FloatTensor of xyz depth images

            @return: fg_logits: a [N x 3 x H x W] torch.FloatTensor of background/table/object logits
                     center_direction_prediction: a [N x 2 x H x W] torch.FloatTensor of center direction predictions
        """
        features = self.decoder([self.encoder(xyz_img)])
        fg_logits = self.fg_module(features)
        center_direction_prediction = self.cd_module(features)

        return fg_logits, center_direction_prediction

class DepthSeedingNetwork(object):

    def __init__(self, data_loader, params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.params = params
        
        # Build network and losses
        self.setup()
        
    def setup(self):
        """ Setup model, losses, optimizers, misc
        """

        # Encoder
        self.encoder = networks.UNet_Encoder(input_channels=3,
                                             feature_dim=self.params['feature_dim'])
        # self.encoder.to(self.device)

        # Decoder
        self.decoder = networks.UNet_Decoder(num_encoders=1, 
                                             feature_dim=self.params['feature_dim'], 
                                             coordconv=self.params['use_coordconv'])
        # self.decoder.to(self.device)

        # A 1x1 conv layer that goes from embedded features to logits for 3 classes: background (0), table (1), objects (2)
        self.foreground_module = nn.Conv2d(self.params['feature_dim'], 3, 
                                           kernel_size=1, stride=1, 
                                           padding=0, bias=False)
        # self.foreground_module.to(self.device)

        # A 1x1 conv layer that goes from embedded features to 2d pixel direction
        self.center_direction_module = nn.Conv2d(self.params['feature_dim'], 2,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=False)
        # self.center_direction_module.to(self.device)

        # Whole model, for nn.DataParallel
        self.model = CombinedDSN(self.encoder, self.decoder, self.foreground_module, self.center_direction_module)
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          self.model = nn.DataParallel(self.model)
        self.model.to(self.device)


        # Hough Voting stuff (this operates on CUDA only)
        self.hough_voting_layer = hv.HoughVoting(skip_pixels=self.params['skip_pixels'], 
                                                 inlier_threshold=self.params['inlier_threshold'], 
                                                 angle_discretization=self.params['angle_discretization'],
                                                 inlier_distance=self.params['inlier_distance'],
                                                 percentage_threshold=self.params['percentage_threshold'],
                                                 object_center_kernel_radius=self.params['object_center_kernel_radius'],
                                                )

        ### Losses ###
        self.foreground_loss = networks.CELossWeighted()
        # self.center_direction_loss = networks.CosineSimilarityLossMasked() # Note: this doesn't need normalized vectors
        self.center_direction_loss = networks.CosineSimilarityLossMasked(weighted=True)
        
        ### Optimizers ###
        self.reset_optimizer(self.params['lr'])
        
        ### Misc ###
        self.epoch_num = 1
        self.iter_num = 1
        self.infos = dict()

    def reset_optimizer(self, lr, momentum=0.9):
        """ Reset optimizer, e.g. if you want to cut learning rate
        """
        parameters_list = []

        # Encoder
        parameters_list += list(self.encoder.parameters())

        # Decoder
        parameters_list += list(self.decoder.parameters())

        # Foreground module
        parameters_list += list(self.foreground_module.parameters())

        # Center direction module
        parameters_list += list(self.center_direction_module.parameters())

        # Optimizer
        self.optimizer = torch.optim.SGD(parameters_list, lr, momentum=momentum)

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0: # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def train_mode(self):
        """ Put all modules into train mode
        """
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        """ Put all modules into eval mode
        """
        self.encoder.eval()
        self.decoder.eval()

    def extract_features(self, input_dict):
        """ Extract features from input. Run encoders and decoder of either tabletop/transform stage

            @param input_dict: A dictionary of torch tensors of different modalities.
                               e.g. keys could include: rgb, xyz
            @return: A [N x C x H x W] torch tensor of features
        """
        return self.decoder([self.encoder(input_dict['xyz'])])

    def predict_foreground(self, features):
        """ Run the foreground module on features

            @param features: a [N x C x H x W] torch tensor of computed features

            @return: a [N x 3 x H x W] torch tensor of background/table/foreground logits
        """
        return self.foreground_module(features)

    def predict_center_directions(self, features):
        """ Run the center direction prediction module on features

            @param features: a [N x C x H x W] torch tensor of computed features

            @return: a [N x 2 x H x W] torch tensor of foreground logits
        """
        return self.center_direction_module(features)

    def train_epoch(self, curr_epoch, total_epochs, tqdm_file=None):
        """ Runs 1 epoch of training
        """

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        losses = util_.AverageMeter()
        fg_losses = util_.AverageMeter()
        direction_losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.train_mode()

        progress = tqdm(self.data_loader, file=tqdm_file)
        progress.set_description("Ep {0}. {1}/{2}".format(self.epoch_num, curr_epoch, total_epochs))
        for i, batch in enumerate(progress):

            if self.iter_num >= self.params['max_iters']:
                print("Reached maximum number of iterations...")
                break

            # Send everything to GPU
            self.send_batch_to_device(batch)

            # Get labels
            foreground_labels = batch['foreground_labels'] # Shape: [N x H x W]
            direction_labels = batch['direction_labels'] # Shape: [N x 2 x H x W]

            # measure data loading time
            data_time.update(time() - end)
            N, H, W = foreground_labels.shape

            # This is (potentiall) in parallel
            fg_logits, center_direction_prediction = self.model(batch['xyz'])

            ### Tabletop Foreground Loss ###
            fg_masks = foreground_labels.clamp(0,2).long()
            fg_loss = self.foreground_loss(fg_logits, fg_masks)

            ### Center Prediction Loss ###
            # center_direction_prediction = self.center_direction_module(features) # Shape: [N x 2 x H x W]
            direction_loss = self.center_direction_loss(center_direction_prediction, direction_labels, foreground_labels)

            # Total loss. Note: foreground loss is always computed/backpropagated
            loss = self.params['lambda_fg'] * fg_loss + self.params['lambda_direction'] * direction_loss


            ### Gradient descent ###
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), N)
            fg_losses.update(fg_loss.item(), N)
            direction_losses.update(direction_loss.item(), N)

            # Record some information about this iteration
            batch_time.update(time() - end)
            end = time()

            # Record information every x iterations
            if self.iter_num % self.params['iter_collect'] == 0:
                info = {'iter_num': self.iter_num,
                        'Batch Time': round(batch_time.avg, 3),
                        'Data Time': round(data_time.avg, 3),
                        'loss': round(losses.avg, 7),
                        'FG loss': round(fg_losses.avg, 7),
                        'Direction loss': round(direction_losses.avg, 7),
                        }
                self.infos[self.iter_num] = info

                # Reset meters
                batch_time = util_.AverageMeter()
                data_time = util_.AverageMeter()
                losses = util_.AverageMeter()
                fg_losses = util_.AverageMeter()
                direction_losses = util_.AverageMeter()
                end = time()

            self.iter_num += 1

        self.epoch_num += 1

    def train(self, num_epochs, tqdm_file=None):
        """ Run the training
        """
        if tqdm_file is not None:
            tqdm_file = open(tqdm_file, 'w')
        for epoch_iter in range(num_epochs):
            self.train_epoch(epoch_iter+1, num_epochs, tqdm_file=tqdm_file)
            if self.iter_num >= self.params['max_iters']:
                break
        if tqdm_file is not None:
            tqdm_file.close()

    #@profile
    def run_on_batch(self, batch):
        """ Run algorithm on batch of images in eval mode

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor

            @return fg_mask: a [N x H x W] torch.LongTensor with values in {0, 1, 2}
                    center_direction_prediction: a [N x 2 x H x W] torch.FloatTensor
                    object_centers: a list of [2 x num_objects] torch.IntTensor. This list has length N
                    initial_maks: a [N x H x W] torch.IntTensor
        """

        self.eval_mode()
        self.send_batch_to_device(batch)

        with torch.no_grad():

            # Extract features
            features = self.extract_features(batch) # Shape: [N x C x H x W]

            # Foreground
            fg_logits = self.predict_foreground(features) # Shape: [N x 3 x H x W]
            fg_probs = F.softmax(fg_logits, dim=1) # Shape: [N x 3 x H x W]
            fg_mask = torch.argmax(fg_probs, dim=1) # Shape: [N x H x W]

            # Center direction
            center_direction_prediction = self.center_direction_module(features) # Shape: [N x 2 x H x W]
            center_direction_prediction = center_direction_prediction / torch.norm(center_direction_prediction, 
                                                                                   dim=1, 
                                                                                   keepdim=True
                                                                                  ).clamp(min=1e-10)

            initial_masks, num_objects, object_centers_padded = \
                self.hough_voting_layer((fg_mask == OBJECTS_LABEL).int(), 
                                        center_direction_prediction)

        # Compute list of object centers
        width = initial_masks.shape[2]
        object_centers = []
        for i in range(initial_masks.shape[0]):
            object_centers.append(object_centers_padded[i, :, :num_objects[i]])

        return fg_mask, center_direction_prediction, object_centers, initial_masks

    def save(self, name=None, save_dir=None):
        """ Save the model as a checkpoint
        """

        # Save main parameter weights / things
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['encoder'] = self.encoder.state_dict()
        checkpoint['decoder'] = self.decoder.state_dict()
        checkpoint['foreground_module'] = self.foreground_module.state_dict()
        checkpoint['center_direction_module'] = self.center_direction_module.state_dict()

        if save_dir is None:
            save_dir = '/home/chrisxie/projects/ssc/checkpoints/'
        if name is None:
            dataset_name = self.data_loader.dataset.name
            filename = save_dir + 'DepthSeedingNetwork_iter' + str(self.iter_num) \
                                + '_' + dataset_name \
                                + '_' + str(self.params['feature_dim']) + 'c' \
                                + '_checkpoint.pth.tar'
        else:
            filename = save_dir + name + '_checkpoint.pth.tar'
        torch.save(checkpoint, filename)

    def load(self, filename):
        """ Load the model checkpoint
        """
        checkpoint = torch.load(filename)

        if 'encoder' in checkpoint:
            print("Loading depth encoder...")
            self.encoder.load_state_dict(checkpoint['encoder'])
        if 'decoder' in checkpoint:
            print("Loading depth decoder...")
            self.decoder.load_state_dict(checkpoint['decoder'])
        if 'foreground_module' in checkpoint:
            print("Loading foreground module...")
            self.foreground_module.load_state_dict(checkpoint['foreground_module'])
        if 'center_direction_module' in checkpoint:
            print("Loading center direction module...")
            self.center_direction_module.load_state_dict(checkpoint['center_direction_module'])

        # Legacy code
        if 'depth_encoder' in checkpoint:
            print("Loading depth encoder...")
            self.encoder.load_state_dict(checkpoint['depth_encoder'])
        if 'depth_decoder' in checkpoint:
            print("Loading depth decoder...")
            self.decoder.load_state_dict(checkpoint['depth_decoder'])

        # Other stuff
        self.iter_num = checkpoint['iter_num']
        self.epoch_num = checkpoint['epoch_num']
        self.infos = checkpoint['infos']


class RGBRefinementNetwork(object):

    def __init__(self, data_loader, params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.params = params
        
        # Build network and losses
        self.setup()
        
    def setup(self):
        """ Setup model, losses, optimizers, misc
        """

        # Encoder
        self.encoder = networks.UNet_Encoder(input_channels=4,
                                             feature_dim=self.params['feature_dim'])
        self.encoder.to(self.device)

        # Decoder
        self.decoder = networks.UNet_Decoder(num_encoders=1, 
                                             feature_dim=self.params['feature_dim'], 
                                             coordconv=self.params['use_coordconv'])
        self.decoder.to(self.device)

        # A 1x1 conv layer that goes from embedded features to logits for foreground
        self.foreground_module = nn.Conv2d(self.params['feature_dim'], 1, 
                                           kernel_size=1, stride=1, 
                                           padding=0, bias=False)
        self.foreground_module.to(self.device)

        ### Losses ###
        self.foreground_loss = networks.BCEWithLogitsLossWeighted(weighted=True)
        
        ### Optimizers ###
        self.reset_optimizer(self.params['lr'])
        
        ### Misc ###
        self.epoch_num = 1
        self.iter_num = 1
        self.infos = dict()

    def reset_optimizer(self, lr, momentum=0.9):
        """ Reset optimizer, e.g. if you want to cut learning rate
        """
        parameters_list = []

        # Encoder
        parameters_list += list(self.encoder.parameters())

        # Decoder
        parameters_list += list(self.decoder.parameters())

        # Foreground module
        parameters_list += list(self.foreground_module.parameters())

        # Optimizer
        self.optimizer = torch.optim.SGD(parameters_list, lr, momentum=momentum)

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0: # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def train_mode(self):
        """ Put all modules into train mode
        """
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        """ Put all modules into eval mode
        """
        self.encoder.eval()
        self.decoder.eval()

    def extract_features(self, input_dict):
        """ Extract features from input. Run encoders and decoder of either tabletop/transform stage

            @param input_dict: A dictionary of torch tensors of different modalities.
                               e.g. keys could include: rgb, xyz
            @return: A [N x C x H x W] torch tensor of features
        """
        #TODO: concatenate RGB and initial mask
        rgb = input_dict['rgb'] # Shape: [N x 3 x H x W], where H = W = 224
        initial_masks = input_dict['initial_masks'].unsqueeze(1) # Shape: [N x 1 x H x W]
        network_input = torch.cat([rgb, initial_masks], dim=1) # Shape: [N x 4 x H x W]
        return self.decoder([self.encoder(network_input)])

    def predict_foreground(self, features):
        """ Run the foreground module on features

            @param features: a [N x C x H x W] torch tensor of computed features

            @return: a [N x H x W] torch tensor of foreground logits
        """
        return self.foreground_module(features)[:,0,:,:]

    def train_epoch(self, curr_epoch, total_epochs, tqdm_file=None):
        """ Runs 1 epoch of training
        """

        # Some stuff to keep track of
        batch_time = util_.AverageMeter()
        data_time = util_.AverageMeter()
        losses = util_.AverageMeter()
        end = time()

        # Training mode
        self.train_mode()

        progress = tqdm(self.data_loader, file=tqdm_file)
        progress.set_description("Ep {0}. {1}/{2}".format(self.epoch_num, curr_epoch, total_epochs))
        for i, batch in enumerate(progress):

            if self.iter_num >= self.params['max_iters']:
                print("Reached maximum number of iterations...")
                break

            # Send everything to GPU
            self.send_batch_to_device(batch)

            # Get labels
            labels = batch['labels'].float() # Shape: [N x H x W]

            # measure data loading time
            data_time.update(time() - end)
            N, H, W = labels.shape

            # Extract features
            features = self.extract_features(batch) # Shape: [N x C x H x W]

            # Foreground Loss
            logits = self.predict_foreground(features) # Shape: [N x 3 x H x W]
            loss = self.foreground_loss(logits, labels)


            ### Gradient descent ###
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), N)

            # Record some information about this iteration
            batch_time.update(time() - end)
            end = time()

            # Record information every x iterations
            if self.iter_num % self.params['iter_collect'] == 0:
                info = {'iter_num': self.iter_num,
                        'Batch Time': round(batch_time.avg, 3),
                        'Data Time': round(data_time.avg, 3),
                        'loss': round(losses.avg, 7),
                        }
                self.infos[self.iter_num] = info

                # Reset meters
                batch_time = util_.AverageMeter()
                data_time = util_.AverageMeter()
                losses = util_.AverageMeter()
                end = time()

            self.iter_num += 1

        self.epoch_num += 1

    def train(self, num_epochs, tqdm_file=None):
        """ Run the training
        """
        if tqdm_file is not None:
            tqdm_file = open(tqdm_file, 'w')
        for epoch_iter in range(num_epochs):
            self.train_epoch(epoch_iter+1, num_epochs, tqdm_file=tqdm_file)
            if self.iter_num >= self.params['max_iters']:
                break
        if tqdm_file is not None:
            tqdm_file.close()

    def run_on_batch(self, batch, threshold=0.5):
        """ Run algorithm on batch of images in eval mode

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - initial_masks: a [N x H x W] torch.FloatTensor
        """

        self.eval_mode()
        self.send_batch_to_device(batch)

        with torch.no_grad():

            features = self.extract_features(batch) # Shape: [N x C x H x W]
            logits = self.predict_foreground(features) # Shape: [N x H x W]
            probs = torch.sigmoid(logits) # Shape: [N x H x W]
            masks = probs > threshold 

        return masks

    def save(self, name=None, save_dir=None):
        """ Save the model as a checkpoint
        """

        # Save main parameter weights / things
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['encoder'] = self.encoder.state_dict()
        checkpoint['decoder'] = self.decoder.state_dict()
        checkpoint['foreground_module'] = self.foreground_module.state_dict()

        if save_dir is None:
            save_dir = '/home/chrisxie/projects/ssc/checkpoints/'
        if name is None:
            dataset_name = self.data_loader.dataset.name + '_' + self.data_loader.dataset.params['starts_file'].replace('.txt', '')
            filename = save_dir + 'RGBRefinementNetwork_iter' + str(self.iter_num) \
                                + '_' + dataset_name \
                                + '_' + str(self.params['feature_dim']) + 'c' \
                                + '_checkpoint.pth.tar'
        else:
            filename = save_dir + name + '_checkpoint.pth.tar'
        torch.save(checkpoint, filename)

    def load(self, filename):
        """ Load the model checkpoint
        """
        checkpoint = torch.load(filename)

        if 'encoder' in checkpoint:
            print("Loading rgb encoder...")
            self.encoder.load_state_dict(checkpoint['encoder'])
        if 'decoder' in checkpoint:
            print("Loading rgb decoder...")
            self.decoder.load_state_dict(checkpoint['decoder'])
        if 'foreground_module' in checkpoint:
            print("Loading foreground module...")
            self.foreground_module.load_state_dict(checkpoint['foreground_module'])

        # Other stuff
        self.iter_num = checkpoint['iter_num']
        self.epoch_num = checkpoint['epoch_num']
        self.infos = checkpoint['infos']


class TableTopSegmentor(object):
    """ Class to encapsulate both Depth Seeding Network and RGB Refinement Network

        There is NO training in this class
    """

    def __init__(self, params, dsn_filename, dsn_params, rrn_filename, rrn_params):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.depth_seeding_network = DepthSeedingNetwork(None, dsn_params) # Set data loader to be None
        self.depth_seeding_network.load(dsn_filename)

        self.rgb_refinement_network = RGBRefinementNetwork(None, rrn_params) # Set data loader to be None
        self.rgb_refinement_network.load(rrn_filename)

        self.params = params

    def process_initial_masks(self, batch, initial_masks, object_centers, fg_masks):
        """ Process the initial masks:
                - open/close morphological transform
                - closest connected component to object center
                - RANSAC to estimate tables to get rid of false positive object masks

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in [0, 2, 3, ...]. No table
            @param object_centers: a list of [2 x num_objects] torch.IntTensor. This list has length N
            @param fg_masks: a [N x H x W] torch.LongTensor with values in {0, 1, 2}
        """
        N, H, W = initial_masks.shape

        # Bring some tensors to numpy for processing
        initial_masks = initial_masks.cpu().numpy()
        for i in range(N):
            object_centers[i] = object_centers[i].cpu().numpy()
        fg_masks = fg_masks.cpu().numpy()
        xyz_imgs = batch['xyz'].cpu().numpy().transpose(0,2,3,1) # Shape: [N x H x W x 3]

        # Open/close morphology stuff
        if self.params['use_open_close_morphology']:

            for i in range(N):

                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # For each object id, open/close the masks
                for obj_id in obj_ids:
                    mask = (initial_masks[i] == obj_id) # Shape: [H x W]

                    ksize = self.params['open_close_morphology_ksize'] # 9
                    opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                                   cv2.MORPH_OPEN, 
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
                    opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                          cv2.MORPH_CLOSE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    h_idx, w_idx = np.nonzero(mask)
                    initial_masks[i, h_idx, w_idx] = 0
                    h_idx, w_idx = np.nonzero(opened_closed_mask)
                    initial_masks[i, h_idx, w_idx] = obj_id

        # Closest Connected Component
        if self.params['use_closest_connected_component']:

            pixel_indices = util_.build_matrix_of_indices(H, W)
            for i in range(N):
                
                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # Loop over each object
                for obj_index, obj_id in enumerate(obj_ids):

                    # Run connected components algorithm
                    components, num_components = connected_components(initial_masks[i] == obj_id)
                    obj_center = object_centers[i][:, obj_index] # y, x location

                    # Find closest connected component via set distance
                    closest_component_num = -1
                    closest_component_dist = 1e10 # start with something ridiculously large
                    for j in range(1, num_components+1):
                        h_idx, w_idx = np.where(components == j)
                        dist = np.linalg.norm(pixel_indices[h_idx, w_idx, :] - obj_center, axis=1).min() # set distance
                        if dist < closest_component_dist:
                            closest_component_num = j
                            closest_component_dist = dist

                    # Fix the initial mask for this object
                    initial_masks[i][initial_masks[i] == obj_id] = 0
                    initial_masks[i][components == closest_component_num] = obj_id


        # RANSAC to estimate table plane from predicted table mask, and get rid of false positive masks
        plane_masks = np.zeros_like(initial_masks)
        distance_from_table = np.zeros((N,H,W), dtype=np.float32)
        if self.params['table_RANSAC']:

            for i in range(N):

                # Get predicted table points
                table_mask = (fg_masks[i, ...] == TABLE_LABEL)
                predicted_table_points = xyz_imgs[i, table_mask, :] # Shape: [num_pixels x 3], of [x, y, z] coordinates

                if np.count_nonzero(table_mask) == 0: # No table
                    continue

                # Uses the equation: ax + by + cz + d = 0, with c = 1.
                xy = predicted_table_points[:, :2] # Shape: [num_pixels x 2]
                neg_z = - predicted_table_points[:, 2] # Shape: [num_pixels]

                # Run the RANSAC algorithm
                ransac = sklearn.linear_model.RANSACRegressor(min_samples=self.params['RANSAC_min_samples'],
                                                              residual_threshold=self.params['RANSAC_residual_threshold'],
                                                              max_trials=self.params['RANSAC_max_trials'],
                                                             )
                ransac.fit(xy, neg_z)

                # Compute inlier mask from entire point cloud. Here, we use the RANSAC threshold in units of meters
                w = np.append(ransac.estimator_.coef_, 1); # table normal vector
                w_norm = np.linalg.norm(w); w = w / w_norm;
                d = ransac.estimator_.intercept_; d = d / w_norm
                if w.dot([0,1,0]) < 0: # Flip the normal so it points towards the camera
                    w = -w; d = -d
                distance = xyz_imgs[i,...].reshape(-1,3).dot(w) + d # this is in units of meters
                distance = distance.reshape(H,W)
                distance_from_table[i,...] = distance
                # Note: this is almost equivalent to computing (ransac.estimator_.predict(xy) - neg_z) / w_norm, except I flipped the sign so the normal vector points upwards

                plane_mask = abs(distance) < ransac.residual_threshold
                plane_masks[i] = plane_mask # Save it

                # Remove false positive masks
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]
                for obj_id in obj_ids:
                    mask = (initial_masks[i] == obj_id) # Shape: [H x W]
                    percent_on_table = np.sum(np.logical_and(mask, plane_mask)) / (np.sum(mask) + 1e-10)
                    if percent_on_table > self.params['false_positive_table_percentage_threshold']:
                        # print(i, percent_on_table)
                        h_idx, w_idx = np.nonzero(mask)
                        initial_masks[i, h_idx, w_idx] = 0



        # Put initial masks back to GPU
        initial_masks = torch.from_numpy(initial_masks).to(self.device)
        plane_masks = torch.from_numpy(plane_masks).to(self.device)
        distance_from_table = torch.from_numpy(distance_from_table).to(self.device)

        return initial_masks, plane_masks, distance_from_table


    def run_on_batch(self, batch):
        """ Run algorithm on batch of images in eval mode

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param final_close_morphology: If True, then run open/close morphology after refining mask.
                                           This typically helps a synthetically-trained RRN
        """
        N, _, H, W = batch['rgb'].shape


        # Run the Depth Seeding Network. Note: this will send "batch" to device (e.g. GPU)
        fg_masks, direction_predictions, object_centers, initial_masks = self.depth_seeding_network.run_on_batch(batch)
        # fg_masks: a [N x H x W] torch.LongTensor with values in {0, 1, 2}
        # direction_predictions: a [N x 2 x H x W] torch.FloatTensor
        # object_centers: a list of [2 x num_objects] torch.IntTensor. This list has length N
        # initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in [0, 2, 3, ...]. No table

        initial_masks, plane_masks, distance_from_table = self.process_initial_masks(batch, 
                                                                                     initial_masks, 
                                                                                     object_centers, 
                                                                                     fg_masks
                                                                                    )

        # Data structure to hold everything at end
        refined_masks = torch.zeros_like(initial_masks)
        for i in range(N):

            # Dictionary to save crop indices
            crop_indices = {}

            mask_ids = torch.unique(initial_masks[i])
            if mask_ids[0] == 0:
                mask_ids = mask_ids[1:]
            rgb_crops = torch.zeros((mask_ids.shape[0], 3, 224, 224), device=self.device)
            mask_crops = torch.zeros((mask_ids.shape[0], 224, 224), device=self.device)

            for index, mask_id in enumerate(mask_ids):
                mask = (initial_masks[i] == mask_id).float() # Shape: [H x W]

                # crop the masks/rgb to 224x224 with some padding, save it as "initial_masks"
                x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)
                x_padding = torch.round((x_max - x_min).float() * self.params['padding_percentage']).item()
                y_padding = torch.round((y_max - y_min).float() * self.params['padding_percentage']).item()

                # Pad and be careful of boundaries
                x_min = max(x_min - x_padding, 0)
                x_max = min(x_max + x_padding, W-1)
                y_min = max(y_min - y_padding, 0)
                y_max = min(y_max + y_padding, H-1)
                crop_indices[mask_id.item()] = [x_min, y_min, x_max, y_max] # save crop indices

                # Crop
                rgb_crop = batch['rgb'][i, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
                mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]

                # Resize
                new_size = (224,224)
                rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
                rgb_crops[index] = rgb_crop
                mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
                mask_crops[index] = mask_crop

            # Run the RGB Refinement Network
            if mask_ids.shape[0] > 0: # only run if you actually have masks to refine...

                new_batch = {'rgb' : rgb_crops, 'initial_masks' : mask_crops}
                refined_crops = self.rgb_refinement_network.run_on_batch(new_batch) # Shape: [num_masks x new_H x new_W]

            # resize the results to the original size. Order this by average depth (highest to lowest)
            sorted_mask_ids = []
            for index, mask_id in enumerate(mask_ids):

                # Resize back to original size
                x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
                orig_H = y_max - y_min + 1
                orig_W = x_max - x_min + 1
                mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
                resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]
                if tuple(resized_mask.size())==(1,1):
                    continue
                # Calculate average depth
                try:
                    h_idx, w_idx = torch.nonzero(resized_mask).t()
                    avg_depth = torch.mean(batch['xyz'][i, 2, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx])
                    sorted_mask_ids.append((index, mask_id, avg_depth))
                except:
                    print('SSC error!')
                

            sorted_mask_ids = sorted(sorted_mask_ids, key=lambda x : x[2], reverse=True)
            sorted_mask_ids = [x[:2] for x in sorted_mask_ids] # list of tuples: (index, mask_id)

            for index, mask_id in sorted_mask_ids:

                # Resize back to original size
                x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
                orig_H = y_max - y_min + 1
                orig_W = x_max - x_min + 1
                mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
                resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

                # Set refined mask
                h_idx, w_idx = torch.nonzero(resized_mask).t()
                refined_masks[i, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = mask_id


        # Open/close morphology stuff, for synthetically-trained RRN
        if self.params['final_close_morphology']:
            refined_masks = refined_masks.cpu().numpy() # to CPU

            for i in range(N):

                # Get object ids. Remove background (0)
                obj_ids = np.unique(refined_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # For each object id, open/close the masks
                for obj_id in obj_ids:
                    mask = (refined_masks[i] == obj_id) # Shape: [H x W]

                    ksize = self.params['open_close_morphology_ksize'] # 9
                    opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                                   cv2.MORPH_OPEN, 
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
                    opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                          cv2.MORPH_CLOSE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    h_idx, w_idx = np.nonzero(mask)
                    refined_masks[i, h_idx, w_idx] = 0
                    h_idx, w_idx = np.nonzero(opened_closed_mask)
                    refined_masks[i, h_idx, w_idx] = obj_id

            refined_masks = torch.from_numpy(refined_masks).to(self.device) # back to GPU

        return fg_masks, direction_predictions, initial_masks, plane_masks, distance_from_table, refined_masks


    def evaluate(self, dl, save_dir):
        """ Evaluate the model on a dataset, save the results

            @param dl: DataLoader for test set
            @param return_annotations: True if you want annotations
        """

        if not save_dir.endswith('/'):
            save_dir = save_dir + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir.replace('RRN_v5_', '')):
            os.makedirs(save_dir.replace('RRN_v5_', ''))

        progress = tqdm(dl)
        for batch in progress:

            # Run model
            fg_masks, direction_predictions, initial_masks, plane_masks, distance_from_table, seg_masks = self.run_on_batch(batch)

            seg_masks = seg_masks.cpu().numpy()
            initial_masks = initial_masks.cpu().numpy()

            # Write results to disk
            for i, path in enumerate(batch['label_abs_path']):
                file_path = save_dir + path.rsplit('/', 1)[0] + '/'
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name = file_path + path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.png'
                util_.imwrite_indexed(file_name, seg_masks[i].astype(np.uint8))

                # also write un-refined version to disk
                # if not os.path.exists(file_path.replace('RRN_v5_', '')):
                #     os.makedirs(file_path.replace('RRN_v5_', ''))
                # unrefined_file_name = file_name.replace('RRN_v5_', '')
                # util_.imwrite_indexed(unrefined_file_name, initial_masks[i].astype(np.uint8))

