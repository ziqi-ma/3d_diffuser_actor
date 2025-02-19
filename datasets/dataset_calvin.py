from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path

import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .dataset_engine import RLBenchDataset
from .utils import Resize, TrajectoryInterpolator
from utils.utils_with_calvin import to_relative_action, convert_rotation
matplotlib.use('Agg') 

# Add this class for drawing 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)


class CalvinDataset(RLBenchDataset):

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=True
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length
            )

        # Keep variations and useful instructions
        self._instructions = instructions
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per-task and variation
        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            self._episodes += eps
            self._num_episodes += len(eps)

        print(f"Created dataset from {root} with {self._num_episodes}")

    def visualize_trajectory(self, episode_id):
        """Visualize keyframes, interpolated trajectory and chunks."""
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]
        episode = self.read_from_cache(file)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all keyframe positions and orientations
        keyframe_positions = []
        keyframe_orientations = []
        for i in episode[0]:  # episode[0] contains frame indices
            pos = episode[2][i][:3]  # Position
            if isinstance(pos, torch.Tensor):
                pos = pos.detach().cpu().numpy()
            else:
                pos = np.array(pos)
            
            quat = episode[2][i][3:7]  # Quaternion
            if isinstance(quat, torch.Tensor):
                quat = quat.detach().cpu().numpy()
            else:
                quat = np.array(quat)
            
            keyframe_positions.append(pos.reshape(-1))
            keyframe_orientations.append(quat.reshape(-1))
        
        keyframe_positions = np.array(keyframe_positions)
        keyframe_orientations = np.array(keyframe_orientations)
        
        # Define distinct colors for each chunk
        chunk_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        chunk_size = self._max_episode_length
        num_chunks = math.ceil(len(keyframe_positions) / chunk_size)
        
        # Plot chunks with different colors and add separators
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(keyframe_positions))
            chunk = keyframe_positions[start_idx:end_idx]
            color = chunk_colors[chunk_idx % len(chunk_colors)]
            
            # Plot keyframes for this chunk
            ax.scatter(chunk[:, 0], chunk[:, 1], chunk[:, 2],
                      c=color, s=100, 
                      label=f'Chunk {chunk_idx + 1} Keyframes')
            
            # Plot interpolated trajectories within this chunk
            for i in range(len(chunk)-1):
                if len(episode) > 5:
                    traj = self._interpolate_traj(episode[5][start_idx + i])
                    if isinstance(traj, torch.Tensor):
                        traj = traj.detach().cpu().numpy()
                    traj_pos = traj[:, :3]
                    ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
                           color=color, linestyle='-', alpha=0.3)
            
            # Draw orientation arrows for keyframes in this chunk
            arrow_length = 0.05
            for pos, quat in zip(chunk, keyframe_orientations[start_idx:end_idx]):
                direction = np.array([1, 0, 0]) * arrow_length
                arrow = Arrow3D((pos[0], pos[0] + direction[0]),
                              (pos[1], pos[1] + direction[1]),
                              (pos[2], pos[2] + direction[2]),
                              mutation_scale=10, lw=1, arrowstyle='->', color=color)
                ax.add_artist(arrow)
            
            # Add a vertical line separator between chunks (if not the last chunk)
            if chunk_idx < num_chunks - 1 and len(chunk) > 0:
                last_pos = chunk[-1]
                next_pos = keyframe_positions[end_idx] if end_idx < len(keyframe_positions) else chunk[-1]
                mid_point = (last_pos + next_pos) / 2
                
                # Draw a vertical separator
                height = 0.1  # Adjust this value to change separator height
                ax.plot([mid_point[0], mid_point[0]], 
                       [mid_point[1], mid_point[1]], 
                       [mid_point[2]-height, mid_point[2]+height],
                       'k--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'Task: {task}, Variation: {variation}')
        
        # Adjust layout to prevent legend from being cut off
        plt.tight_layout()
        
        # Save plot instead of displaying
        save_path = f'chunk_viz/trajectory_vis_{task}_{variation}_{episode_id}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        if episode_id % 20 == 0:  # Visualize every 20th episode
            self.visualize_trajectory(episode_id)

        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0, :, 20:180, 20:180]
        pcds = states[:, :, 1, :, 20:180, 20:180]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        if self._instructions is not None:
            instr_ind = episode[6][0]
            #print(f"instruction type {type(self._instructions)}")
            instr = torch.as_tensor(self._instructions[instr_ind])
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        if len(episode) > 7:
            gripper_history = torch.cat([
                episode[7][i] for i in frame_ids
            ], dim=0)
        else:
            gripper_history = torch.stack([
                torch.cat([episode[4][max(0, i-2)] for i in frame_ids]),
                torch.cat([episode[4][max(0, i-1)] for i in frame_ids]),
                gripper
            ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, traj_items[0].shape[-1])
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        # Compute relative action
        if self._relative_action and traj is not None:
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(to_relative_action(
                        traj[i, j].numpy(), traj[i, 0].numpy(), clip=False
                    ))
            traj = rel_traj

        # Convert Euler angles to Quarternion
        action = torch.cat([
            action[..., :3],
            torch.as_tensor(convert_rotation(action[..., 3:6])),
            action[..., 6:]
        ], dim=-1)
        gripper = torch.cat([
            gripper[..., :3],
            torch.as_tensor(convert_rotation(gripper[..., 3:6])),
            gripper[..., 6:]
        ], dim=-1)
        gripper_history = torch.cat([
            gripper_history[..., :3],
            torch.as_tensor(convert_rotation(gripper_history[..., 3:6])),
            gripper_history[..., 6:]
        ], dim=-1)
        if traj is not None:
            traj = torch.cat([
                traj[..., :3],
                torch.as_tensor(convert_rotation(traj[..., 3:6])),
                traj[..., 6:]
            ], dim=-1)

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })
        return ret_dict
