# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms
import torch
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torch.utils.data import Dataset,random_split,Subset
import pickle

class LINKS_curs_dataset(Dataset):
    """Dataset of curated links"""
    
    def __init__(self,input_data_path: str,bn_data_path: str):
        
        ## saving path details and loading data
        
        self.input_data_path = input_data_path
        self.bn_data_path = bn_data_path
        self.input_data = torch.load(self.input_data_path)
        self.input_data = self.input_data.permute(0,2,1)
        self.bn_data = torch.load(bn_data_path)
        self.num_bodies_data = (self.bn_data[:,102]).long()
        
        self.max_num_bodies = self.num_bodies_data.max() 
        self.min_num_bodies = self.num_bodies_data.min()
        self.unique_num_bodies = self.num_bodies_data.unique() 
        self.count_to_label = {val.item(): idx for (idx,val) in enumerate(self.unique_num_bodies)}
        self.label_to_count = {value:key for (key,value) in self.count_to_label.items()}
        self.num_bodies_labels = torch.tensor([self.count_to_label[val.item()] for val in self.num_bodies_data])
        print(f"There are {self.num_bodies_labels.shape[0]} data points")
    
    def __len__(self):
        return self.num_bodies_labels.shape[0]
    
    def __getitem__(self, idx):
        return self.input_data[idx,...],self.num_bodies_labels[idx]


def get_dataset_save_splits(dataset_class,input_data_path:str,bn_data_path:str,split_pkl_path:str,rng:torch._C.Generator,props:list=[0.8,0.1,0.1]):
    
    all_dataset = dataset_class(input_data_path,bn_data_path)
    nums = [int(all_dataset.__len__()*val) for val in props]
    train_dataset,test_dataset,val_dataset = random_split(all_dataset,props, generator=rng)
    
    # saving the indices
    with open(split_pkl_path,'wb') as f:
        idx_dict = {
            'input_data_path':input_data_path,
            'bn_data_path':bn_data_path,
            'train':train_dataset.indices,
            'val':val_dataset.indices,
            'test':test_dataset.indices
        }
        pickle.dump(idx_dict,f)
    
    return all_dataset,train_dataset,test_dataset,val_dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "link_data":
        
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
