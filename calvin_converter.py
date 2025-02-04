import numpy as np
import cv2
from pathlib import Path
import json
from scipy.spatial.transform import Rotation
import logging
import os

class CALVINConverter:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CALVINConverter")        
        self.GRIPPER_THRESHOLD = 0.01
        self.TRANSLATION_SCALE = 50.0  
        self.ROTATION_SCALE = 20.0 
        
    def load_data(self):
        self.logger.info("Loading source data...")        
        self.ee_states = np.load(self.source_dir / 'testing_demo_ee_states.npz')['data']
        self.joint_states = np.load(self.source_dir / 'testing_demo_joint_states.npz')['data']
        self.gripper_states = np.load(self.source_dir / 'testing_demo_gripper_states.npz')['data']
        self.actions = np.load(self.source_dir / 'testing_demo_action.npz')['data']        
        self.rgb_static = np.load(self.source_dir / 'testing_demo_camera_0_rgba.npz')['data']
        self.depth_static = np.load(self.source_dir / 'testing_demo_camera_0_depth.npz')['data']
        self.camera_info = np.load(self.source_dir / 'testing_demo_camera_0_info.npz', allow_pickle=True)['data']
        print("Camera info structure:", self.camera_info)        
        self.timestamps = np.load(self.source_dir / 'testing_demo_camera_0_timestamp.npz')['data']
        
        with open(self.source_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
            
        self.logger.info("Data loading complete")
        
    def get_robot_indices_for_camera_frames(self):
        robot_timestamps = np.linspace(self.timestamps[0], self.timestamps[-1], len(self.ee_states))
        return np.array([np.argmin(np.abs(robot_timestamps - ts)) for ts in self.timestamps])
        
    def process_camera_data(self):
        """
        RGB Static:
        - Load RGBA (1358, 720, 1280, 4)
        - Drop alpha channel → (1358, 720, 1280, 3)
        - Resize to (1358, 200, 200, 3)
        - Convert to uint8        

        RGB Gripper:
        - Extract frames from rgb_video.mp4
        - Resize to (1358, 84, 84, 3)
        - Convert to uint8

        Depth Static:
        - Load depth (1358, 720, 1280)
        - Resize to (1358, 200, 200)
        - Ensure float32

        Depth Gripper:
        - Extract frames from depth_video.mp4
        - Resize to (1358, 84, 84)
        - Convert to float32                
        """
        self.logger.info("Processing camera data...")        
        self.calvin_rgb_static = []
        self.calvin_depth_static = []
        
        for frame_idx in range(len(self.rgb_static)):
            rgb = self.rgb_static[frame_idx, :, :, :3]
            rgb_resized = cv2.resize(rgb, (200, 200))
            self.calvin_rgb_static.append(rgb_resized)            
            depth = self.depth_static[frame_idx]
            depth_resized = cv2.resize(depth, (200, 200))
            self.calvin_depth_static.append(depth_resized)
            
        self.calvin_rgb_static = np.array(self.calvin_rgb_static, dtype=np.uint8)
        self.calvin_depth_static = np.array(self.calvin_depth_static, dtype=np.float32)        
        rgb_frames = []
        depth_frames = []        
        rgb_cap = cv2.VideoCapture(str(self.source_dir / 'rgb_video.mp4'))
        depth_cap = cv2.VideoCapture(str(self.source_dir / 'depth_video.mp4'))
        
        while True:
            rgb_ret, rgb_frame = rgb_cap.read()
            depth_ret, depth_frame = depth_cap.read()
            
            if not rgb_ret or not depth_ret:
                break                
            rgb_resized = cv2.resize(rgb_frame, (84, 84))
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            depth_resized = cv2.resize(depth_frame, (84, 84))
            rgb_frames.append(rgb_resized)
            depth_frames.append(depth_resized)
        rgb_cap.release()
        depth_cap.release()
        self.calvin_rgb_gripper = np.array(rgb_frames, dtype=np.uint8)
        self.calvin_depth_gripper = np.array(depth_frames, dtype=np.float32)
        
    def process_robot_data(self, robot_indices):
        """
        Robot Actions:
            TCP position:
            - Extract columns 12-14 from ee_states

            TCP orientation:
            - Convert 3x3 rotation matrix (first 9 elements) to euler angles
        
        Gripper:
            - Convert width to binary (-1 if ≤ 0.01, 1 if > 0.01)

        Relative Actions:
        - Compute differences between consecutive absolute actions
        - Scale translation by 50 (per CALVIN docs)
        - Scale rotation by 20 (per CALVIN docs)
        - Clip to [-1, 1]

        Robot Observations:
        - Combine:
            * TCP position (3)
            * TCP orientation (3)
            * Gripper width (1)
            * Joint states (7)
            * Gripper action (1)        
        """
        self.logger.info("Processing robot data...")        
        self.calvin_robot_obs = []
        self.calvin_actions = []
        
        for idx in robot_indices:
            ee_state = self.ee_states[idx]            
            position = ee_state[12:15]            
            rot_matrix = ee_state[:9].reshape(3, 3)
            rot = Rotation.from_matrix(rot_matrix)
            orientation = rot.as_euler('xyz')            
            gripper_binary = -1.0 if self.gripper_states[idx] <= self.GRIPPER_THRESHOLD else 1.0            
            action = np.concatenate([position, orientation, [gripper_binary]])
            self.calvin_actions.append(action)
            
            # (15-DOF)
            robot_obs = np.concatenate([
                position,                    # TCP position (3)
                orientation,                 # TCP orientation (3)
                [self.gripper_states[idx]],  # Gripper width (1)
                self.joint_states[idx],      # Joint states (7)
                [gripper_binary]            # Gripper action (1)
            ])
            self.calvin_robot_obs.append(robot_obs)
        
        self.calvin_actions = np.array(self.calvin_actions, dtype=np.float32)
        self.calvin_robot_obs = np.array(self.calvin_robot_obs, dtype=np.float32)        
        self.calvin_rel_actions = np.zeros_like(self.calvin_actions)
        self.calvin_rel_actions[1:, :3] = np.diff(self.calvin_actions[:, :3], axis=0) * self.TRANSLATION_SCALE
        self.calvin_rel_actions[1:, 3:6] = np.diff(self.calvin_actions[:, 3:6], axis=0) * self.ROTATION_SCALE
        self.calvin_rel_actions[:, 6] = self.calvin_actions[:, 6]  # Keep gripper action as is        
        self.calvin_rel_actions = np.clip(self.calvin_rel_actions, -1, 1)
        
    def save_calvin_format(self):
        self.logger.info("Saving in CALVIN format...")        
        train_dir = self.output_dir / 'training'
        val_dir = self.output_dir / 'validation'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        scene_obs = np.zeros(24, dtype=np.float32) #dummy scene_obs (24,) vector
        n_episodes = len(self.timestamps)
        n_train = int(0.8 * n_episodes)
        camera_params = self.camera_info[0]
        camera_info = {
            'static': {
                'resolution_width': camera_params['resolution_width'],
                'resolution_height': camera_params['resolution_height'],
                'fx': camera_params['fx'],
                'fy': camera_params['fy'],
                'cx': camera_params['cx'],
                'cy': camera_params['cy'],
                'h_fov': camera_params['h_fov'],
                'v_fov': camera_params['v_fov']
            },
            'gripper': {
                'resolution_width': camera_params['resolution_width'],
                'resolution_height': camera_params['resolution_height'],
                'fx': camera_params['fx'],
                'fy': camera_params['fy'],
                'cx': camera_params['cx'],
                'cy': camera_params['cy'],
                'h_fov': camera_params['h_fov'],
                'v_fov': camera_params['v_fov']
            }
        }
        np.savez_compressed(self.output_dir / 'camera_info.npz', data=camera_info)
        
        for i in range(n_episodes):
            episode_data = {
                'rgb_static': self.calvin_rgb_static[i],  # (200, 200, 3)
                'depth_static': self.calvin_depth_static[i],  # (200, 200)
                'rgb_gripper': self.calvin_rgb_gripper[i],  # (84, 84, 3)
                'depth_gripper': self.calvin_depth_gripper[i],  # (84, 84)
                'robot_obs': self.calvin_robot_obs[i],  # (15,)
                'scene_obs': scene_obs,  # (24,)
            }            
            if i < n_train:
                save_path = train_dir / f'episode_{i:07d}.npz'
            else:
                save_path = val_dir / f'episode_{i:07d}.npz'
            
            np.savez_compressed(save_path, **episode_data)
        
        self.logger.info(f"Saved {n_train} training episodes and {n_episodes - n_train} validation episodes")
        
    def convert(self):
        self.load_data()
        robot_indices = self.get_robot_indices_for_camera_frames()
        self.process_camera_data()
        self.process_robot_data(robot_indices)
        self.save_calvin_format()
        self.logger.info("Conversion complete!")

def main():
    converter = CALVINConverter(
        source_dir='run10',
        output_dir='calvin_new'
    )
    converter.convert()

if __name__ == '__main__':
    main()