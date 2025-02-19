from typing import List, Optional
from pathlib import Path
import os
import pickle

import tap
import cv2
import numpy as np
import torch
import blosc
from PIL import Image
from utils.utils_with_calvin import (
    keypoint_discovery,
    deproject,
    get_gripper_camera_view_matrix,
    convert_rotation
)


class Arguments(tap.Tap):
    traj_len: int = 16
    execute_every: int = 4
    save_path: str = 'calvin_complete_processed'
    root_dir: str = 'calvin_complete'
    mode: str = 'keypose'  # [keypose, close_loop]
    tasks: Optional[List[str]] = None
    split: str = 'validation'  # [training, validation]

def process_datas(datas, mode, traj_len, execute_every, keyframe_inds):
    """Fetch and drop datas to make a trajectory

    Args:
        datas: a dict of the datas to be saved/loaded
            - static_pcd: a list of nd.arrays with shape (height, width, 3)
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_pcd: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (7,)
        mode: a string of [keypose, close_loop]
        traj_len: an int of the length of the trajectory
        execute_every: an int of execution frequency
        keyframe_inds: an Integer array with shape (num_keyframes,)

    Returns:
        the episode item: [
            [frame_ids],
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [annotation_ind] # wrt frame_ids, (1,)
        ]
    """
    # upscale gripper camera
    h, w = datas['static_rgb'][0].shape[:2]
    datas['gripper_rgb'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in datas['gripper_rgb']
    ]
    datas['gripper_pcd'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        for m in datas['gripper_pcd']
    ]
    static_rgb = np.stack(datas['static_rgb'], axis=0) # (traj_len, H, W, 3)
    static_pcd = np.stack(datas['static_pcd'], axis=0) # (traj_len, H, W, 3)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0) # (traj_len, H, W, 3)
    gripper_pcd = np.stack(datas['gripper_pcd'], axis=0) # (traj_len, H, W, 3)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1) # (traj_len, ncam, H, W, 3)
    pcd = np.stack([static_pcd, gripper_pcd], axis=1) # (traj_len, ncam, H, W, 3)
    rgb_pcd = np.stack([rgb, pcd], axis=2) # (traj_len, ncam, 2, H, W, 3)])
    rgb_pcd = rgb_pcd.transpose(0, 1, 2, 5, 3, 4) # (traj_len, ncam, 2, 3, H, W)
    rgb_pcd = torch.as_tensor(rgb_pcd, dtype=torch.float32) # (traj_len, ncam, 2, 3, H, W)

    # prepare keypose actions
    keyframe_indices = torch.as_tensor(keyframe_inds)[None, :]
    gripper_indices = torch.arange(len(datas['proprios'])).view(-1, 1)
    action_indices = torch.argmax(
        (gripper_indices < keyframe_indices).float(), dim=1
    ).tolist()
    action_indices[-1] = len(keyframe_inds) - 1
    actions = [datas['proprios'][keyframe_inds[i]] for i in action_indices]
    action_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1) for a in actions
    ]

    # prepare camera_dicts
    camera_dicts = [{'front': (0, 0), 'wrist': (0, 0)}]

    # prepare gripper tensors
    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ]

    # prepare trajectories
    if mode == 'keypose':
        trajectories = []
        for i in range(len(action_indices)):
            target_frame = keyframe_inds[action_indices[i]]
            current_frame = i
            trajectories.append(
                torch.cat(
                    [
                        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                        for a in datas['proprios'][current_frame:target_frame+1]
                    ],
                    dim=0
                )
            )
    else:
        trajectories = []
        for i in range(len(gripper_tensors)):
            traj = datas['proprios'][i:i+traj_len]
            if len(traj) < traj_len:
                traj += [traj[-1]] * (traj_len - len(traj))
            traj = [
                torch.as_tensor(a, dtype=torch.float32).view(1, -1)
                for a in traj
            ]
            traj = torch.cat(traj, dim=0)
            trajectories.append(traj)

    # Filter out datas
    if mode == 'keypose':
        keyframe_inds = [0] + keyframe_inds[:-1].tolist()
        keyframe_indices = torch.as_tensor(keyframe_inds)
        rgb_pcd = torch.index_select(rgb_pcd, 0, keyframe_indices)
        action_tensors = [action_tensors[i] for i in keyframe_inds]
        gripper_tensors = [gripper_tensors[i] for i in keyframe_inds]
        trajectories = [trajectories[i] for i in keyframe_inds]
    else:
        rgb_pcd = rgb_pcd[:-1]
        action_tensors = action_tensors[:-1]
        gripper_tensors = gripper_tensors[:-1]
        trajectories = trajectories[:-1]

        rgb_pcd = rgb_pcd[::execute_every]
        action_tensors = action_tensors[::execute_every]
        gripper_tensors = gripper_tensors[::execute_every]
        trajectories = trajectories[::execute_every]

    # prepare frame_ids
    frame_ids = [i for i in range(len(rgb_pcd))]

    # Save everything to disk
    state_dict = [
        frame_ids,
        rgb_pcd,
        action_tensors,
        camera_dicts,
        gripper_tensors,
        trajectories,
        datas['annotation_id']
    ]
    return state_dict


def load_episode(root_dir, split, episode, datas, ann_id):
    data = np.load(f'{root_dir}/{split}/{episode}')
    camera_info_file = np.load(f'{root_dir}/camera_info.npz', allow_pickle=True)
    camera_info = camera_info_file['data'].item()
    
    # Get camera IDs for static and gripper cameras
    # Static camera has larger ID (33387783), gripper camera has smaller ID (19798856)
    camera_ids = list(camera_info.keys())
    static_cam_id = max(camera_ids)  # Larger ID is static camera
    gripper_cam_id = min(camera_ids)  # Smaller ID is gripper camera
    
    # Create camera parameter dictionaries
    static_cam_params = {
        'resolution_width': camera_info[static_cam_id]['resolution_width'],
        'resolution_height': camera_info[static_cam_id]['resolution_height'],
        'fx': camera_info[static_cam_id]['fx'],
        'fy': camera_info[static_cam_id]['fy'],
        'cx': camera_info[static_cam_id]['cx'],
        'cy': camera_info[static_cam_id]['cy'],
        'width': data['rgb_static'].shape[1],
        'height': data['rgb_static'].shape[0]
    }
    
    gripper_cam_params = {
        'resolution_width': camera_info[gripper_cam_id]['resolution_width'],
        'resolution_height': camera_info[gripper_cam_id]['resolution_height'],
        'fx': camera_info[gripper_cam_id]['fx'],
        'fy': camera_info[gripper_cam_id]['fy'],
        'cx': camera_info[gripper_cam_id]['cx'],
        'cy': camera_info[gripper_cam_id]['cy'],
        'width': data['rgb_gripper'].shape[1],
        'height': data['rgb_gripper'].shape[0]
    }
    
    # Use the modified deproject function
    static_pcd = deproject(
        static_cam_params, 
        data['depth_static'],
        homogeneous=False
    ).transpose(1, 0)
    static_pcd = np.reshape(
        static_pcd, (data['depth_static'].shape[0], data['depth_static'].shape[1], 3)
    )

    gripper_pcd = deproject(
        gripper_cam_params,
        data['depth_gripper'],
        homogeneous=False
    ).transpose(1, 0)
    gripper_pcd = np.reshape(
        gripper_pcd, (data['depth_gripper'].shape[0], data['depth_gripper'].shape[1], 3)
    )

    # map RGB to [-1, 1]
    rgb_static = data['rgb_static'] / 255. * 2 - 1
    rgb_gripper = data['rgb_gripper'] / 255. * 2 - 1

    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        data['robot_obs'][:3],
        data['robot_obs'][3:6],
        (data['robot_obs'][[-1]] > 0).astype(np.float32)
    ], axis=-1)

    # Put them into a dict
    datas['static_pcd'].append(static_pcd)  # (200, 200, 3)
    datas['static_rgb'].append(rgb_static)  # (200, 200, 3)
    datas['gripper_pcd'].append(gripper_pcd)  # (84, 84, 3)
    datas['gripper_rgb'].append(rgb_gripper)  # (84, 84, 3)
    datas['proprios'].append(proprio)  # (8,)
    datas['annotation_id'].append(ann_id)  # int


def init_datas():
    datas = {
        'static_pcd': [],
        'static_rgb': [],
        'gripper_pcd': [],
        'gripper_rgb': [],
        'proprios': [],
        'annotation_id': []
    }
    return datas


def main(split, args):
    """
    Process CALVIN-style dataset using language annotations.
    Each annotation represents one complete task sequence.
    """
    # Load language annotations
    annotations = np.load(
        f'{args.root_dir}/{split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    # Process each task sequence from the annotations
    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        # Skip if task filtering is enabled
        if args.tasks is not None and annotations['language']['task'][anno_ind] not in args.tasks:
            continue
            
        print(f'Processing annotation {anno_ind}, start_id:{start_id}, end_id:{end_id}')
        datas = init_datas()
        
        # Load all episodes for this task sequence
        for ep_id in range(start_id, end_id + 1):
            episode = f'episode_{ep_id:07d}.npz'
            try:
                load_episode(
                    args.root_dir,
                    split,
                    episode,
                    datas,
                    anno_ind
                )
            except FileNotFoundError:
                print(f"Warning: Episode {episode} not found, skipping...")
                continue

        # Detect keyframes within the sequence
        _, keyframe_inds, check = keypoint_discovery(datas['proprios'])

        state_dict = process_datas(
            datas, args.mode, args.traj_len, args.execute_every, keyframe_inds
        )

        # For testing, use simple scene assignment
        scene = 'A' if split == 'training' else 'A'

        # Save processed data
        ep_save_path = f'{args.save_path}/{split}/{scene}+0/ann_{anno_ind}.dat'
        os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
        with open(ep_save_path, "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.split, args)
