"""
Custom dataset loader for DroneRace data in native NeRF/INGP coordinate system
This bypasses the Cambridge-specific coordinate transformations that are causing issues
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import json
import cv2
from dataset_loaders.utils.color import rgb_to_yuv
from dataset_loaders.cambridge_scenes import load_image, load_depth_image

class DroneRace(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, 
                 mode=1, seed=7, df=1, trainskip=1, testskip=1, ret_idx=False, ret_hist=False,
                 hist_bin=10, **kwargs):
        """
        DroneRace dataset loader that preserves native NeRF coordinate system
        """
        self.mode = mode
        self.ret_idx = ret_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.df = df
        
        # Set up paths
        root_dir = osp.join(data_path, scene) + ('/train' if train else '/test')
        rgb_dir = root_dir + '/rgb/'
        pose_dir = root_dir + '/poses/'
        world_setup_fn = osp.join(data_path, scene) + '/world_setup.json'
        
        # Collect RGB files and pose files
        self.rgb_files = os.listdir(rgb_dir)
        self.rgb_files = [rgb_dir + f for f in self.rgb_files]
        self.rgb_files.sort()
        
        self.pose_files = os.listdir(pose_dir)
        self.pose_files = [pose_dir + f for f in self.pose_files]
        self.pose_files.sort()
        
        if len(self.rgb_files) != len(self.pose_files):
            raise Exception('RGB file count does not match pose file count!')
        
        # Read world setup
        with open(world_setup_fn, 'r') as f:
            world_setup = json.load(f)
        self.near = world_setup['near']
        self.far = world_setup['far']
        
        # Apply skipping
        frame_idx = np.arange(len(self.rgb_files))
        if train and trainskip > 1:
            frame_idx = frame_idx[::trainskip]
        elif not train and testskip > 1:
            frame_idx = frame_idx[::testskip]
        
        self.rgb_files = [self.rgb_files[i] for i in frame_idx]
        self.pose_files = [self.pose_files[i] for i in frame_idx]
        
        # Load poses - NO COORDINATE TRANSFORMATIONS APPLIED
        poses = []
        for pose_file in self.pose_files:
            pose = np.loadtxt(pose_file)  # Load 4x4 matrix
            poses.append(pose)
        poses = np.array(poses)  # [N, 4, 4]
        
        # Convert to [N, 3, 4] format and flatten to 12 elements (standard NeRF format)
        self.poses = poses[:, :3, :4].reshape(poses.shape[0], 12)
        
        # Get image dimensions
        img = load_image(self.rgb_files[0])
        img_np = (np.array(img) / 255.).astype(np.float32)
        self.H, self.W = img_np.shape[:2]
        
        if self.df != 1.:
            self.H = int(self.H // self.df)
            self.W = int(self.W // self.df)
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, index):
        img = load_image(self.rgb_files[index])
        pose = self.poses[index]
        
        if self.df != 1.:
            img_np = (np.array(img) / 255.).astype(np.float32)
            dims = (self.W, self.H)
            img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA)
            img = img_half_res
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            pose = self.target_transform(pose)
        
        if self.ret_idx:
            return img, pose, index
        elif self.ret_hist:
            yuv = rgb_to_yuv(img)
            y_img = yuv[0]
            hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.)
            hist = hist/(hist.sum())*100
            hist = torch.round(hist)
            return img, pose, hist
        
        # For NeRF training, always return index as third element
        return img, pose, index


def load_DroneRace_dataloader_NeRF(args):
    """
    Custom dataloader for DroneRace that preserves NeRF coordinate system
    """
    kwargs = {'scene': 'DroneRace', 'data_path': args.datadir.replace('/DroneRace', ''),
              'trainskip': args.trainskip, 'testskip': args.testskip, 'df': args.df}
    
    # No transforms - use identity
    from torchvision import transforms
    data_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
    
    train_set = DroneRace(train=True, transform=data_transform, 
                         target_transform=target_transform, **kwargs)
    val_set = DroneRace(train=False, transform=data_transform,
                       target_transform=target_transform, **kwargs)
    
    # Create dataloaders
    train_dl = data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    val_dl = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    
    # Extract camera parameters (assuming standard NeRF format)
    H, W = train_set.H, train_set.W
    focal = 0.5 * W / np.tan(0.5 * 2.094395160675049)  # From camera_angle_x in transforms.json
    hwf = [H, W, focal]
    
    # For NeRF, we need train/val splits
    i_train = list(range(len(train_set)))
    i_val = list(range(len(val_set)))
    i_test = i_val  # Use val as test
    i_split = [i_train, i_val, i_test]
    
    # Bounds
    bds = np.array([train_set.near, train_set.far])
    
    # No render poses for now
    render_poses = None
    render_img = None
    
    return train_dl, val_dl, hwf, i_split, bds, render_poses, render_img