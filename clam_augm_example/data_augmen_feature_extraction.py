import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
#from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP_Augment_Slide, Whole_Slide_Bag_FP_Augment_Patch
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from PIL import Image
import h5py
import openslide

from data_augmentation.datasets.data_augmentation_dataset import generate_z, CustomTransform


# Enviroment
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Augmentation set-up
#augm_keys 	    = ['brightness', 'contrast', 'saturation', 'hue', 'hflip', 'vflip']
augm_keys 	    = ['brightness', 'contrast', 'saturation', 'hue']
scale           = 1.
augm_range_args = { 'brightness': 	64./255 * scale,
             		'saturation': 	0.25 	* scale,
              		'hue': 		  	0.04 	* scale,
              		'contrast': 	0.75 	* scale}

def save_hdf5(output_path, asset_dict, mode='a'):
	file = h5py.File(output_path, mode)

	for key, val in asset_dict.items():
		data_shape = val.shape
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
			dset[:] = val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis=0)
			dset[-data_shape[0]:] = val  

	file.close()
	return output_path

# Apply random augmentation to a slide (same transform is applied to all patches within a slide)
def compute_w_loader_augm_slide(file_path, output_path, wsi, model,
	feature_dim = 1024, batch_size = 8, verbose = 0, print_every=20, pretrained=True, custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		feature_dim: feature dimension
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
	"""
	
	# generate a random transformations to apply to all patches in the slide
	augm_tf = CustomTransform(augm_keys)
	z 		= generate_z(augm_keys, augm_range_args, bsize = 1, seed=None)

	dataset = Whole_Slide_Bag_FP_Augment_Slide(file_path=file_path, wsi=wsi, pretrained=pretrained, 
									 custom_downsample=custom_downsample, target_patch_size=target_patch_size,
									 augment_transform = augm_tf, augment_parameters = z[0])
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			mini_bs = coords.shape[0]			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, mode=mode)
			mode = 'a'
	
	return output_path



# Apply random augm. to each patch within a slide
def compute_w_loader_augm_patch(file_path, output_path, wsi, model,
	feature_dim = 1024, batch_size = 8, verbose = 0, print_every=20, pretrained=True, custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		feature_dim: feature dimension
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
	"""
	
	# generate a random transformations to apply to all patches in the slide
	augm_tf = CustomTransform(augm_keys)
	dataset = Whole_Slide_Bag_FP_Augment_Patch(file_path=file_path, wsi=wsi, pretrained=pretrained, 
						custom_downsample=custom_downsample, target_patch_size=target_patch_size,
						augment_transform = augm_tf, augment_keys = augm_keys, augment_range_args = augm_range_args)

	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			mini_bs = coords.shape[0]	
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, mode=mode)
			mode = 'a'
	
	return output_path



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--num_augmentations', type=int, default=1, help='How many times to augment the dataset')
parser.add_argument('--patch_augm', default=False, action='store_true', help='Apply random transf. to each patch within slide, vs. default setup apply the same random transf. to all patches within a slide')
parser.add_argument('--augm_count_start', type=int, default=0, help='Starting counter for augmentation. E.g. if augm_0 already exist, you want to start the counter at augm_1')

args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
        

	for bag_candidate_idx in range(total):
		slide_id = str(bags_dataset[bag_candidate_idx]).split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		wsi = openslide.open_slide(slide_file_path)
		
		for t in range(args.augm_count_start,args.num_augmentations):
			
			time_start = time.time()	
                        
			if not args.patch_augm:
                            bag_name_augm = f"{slide_id}_augm_{t}.h5"
                            output_path   = os.path.join(args.feat_dir, 'h5_files', bag_name_augm)
                            output_file_path = compute_w_loader_augm_slide(h5_file_path, output_path, wsi, model = model, 
							    feature_dim = 1024, batch_size = args.batch_size, 
							    verbose = 1, print_every = 20, 
							    custom_downsample=args.custom_downsample, 
							    target_patch_size=args.target_patch_size)
                       
			else:
                            bag_name_augm = f"{slide_id}_augm_{t}p.h5"
                            output_path   = os.path.join(args.feat_dir, 'h5_files', bag_name_augm)
                            output_file_path = compute_w_loader_augm_patch(h5_file_path, output_path, wsi, model = model,
                                                            feature_dim = 1024, batch_size = args.batch_size,
                                                            verbose = 1, print_every = 20,
                                                            custom_downsample=args.custom_downsample,
                                                            target_patch_size=args.target_patch_size)

			time_elapsed = time.time() - time_start
			print('\ncomputing features for {}, augm. number {}, took {} s'.format(output_file_path, t, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)


			if not args.patch_augm:
			    bag_output_name = f"{bag_base}_augm_{t}.pt"
                       
			else:
			    bag_output_name = f"{bag_base}_augm_{t}p.pt"

			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_output_name))


