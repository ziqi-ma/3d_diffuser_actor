import numpy as np
from scipy.signal import argrelextrema
import torch
import utils.pytorch3d_transforms as pytorch3d_transforms
import pybullet as pb

from calvin_env.robot.robot import Robot
from calvin_env.utils.utils import angle_between_angles

# Global statistics
stats = {
    'total_sequences': 0,
    'sequences_with_maxima': 0,
    'total_maxima_found': 0,
    'total_maxima_used': 0
}

def get_eef_velocity_from_robot(robot: Robot):
    eef_vel = []
    for i in range(2):
        eef_vel.append(
            pb.getJointState(
                robot.robot_uid,
                robot.gripper_joint_ids[i],
                physicsClientId=robot.cid
            )[1]
        )

    # mean over the two gripper points.
    vel = sum(eef_vel) / len(eef_vel)

    return vel


def get_eef_velocity_from_trajectories(trajectories):
    trajectories = np.stack([trajectories[0]] + trajectories, axis=0)
    velocities = trajectories[1:] - trajectories[:-1]

    V = np.linalg.norm(velocities[:, :3], axis=-1)
    W = np.linalg.norm(velocities[:, 3:6], axis=-1)

    velocities = np.concatenate(
        [velocities, [velocities[-1]]],
        # [velocities[[0]], velocities],
        axis=0
    )
    accelerations = velocities[1:] - velocities[:-1]

    A = np.linalg.norm(accelerations[:, :3], axis=-1)

    return V, W, A


def scene_state_changes(scene_states, task):
    """Return the delta of objects in the scene.

    Args:
        scene_states: A list of scene_obs arrays.
            Each array is 24-dimensional:
                sliding door (1): joint state
                drawer (1): joint state
                button (1): joint state
                switch (1): joint state
                lightbulb (1): on=1, off=0
                green light (1): on=1, off=0
                red block (6): (x, y, z, euler_x, euler_y, euler_z)
                blue block (6): (x, y, z, euler_x, euler_y, euler_z)
                pink block (6): (x, y, z, euler_x, euler_y, euler_z)

    Returns:
        An binary array of shape (batch_size, 24) where `1` denotes
        significant state change for the object state.
    """
    all_changed_inds = []
    # For lightbul/green light, we select frames when the light turns on/off.
    if "lightbulb" in task or "switch" in task:
        obj_inds = [4]
    elif "led" in task or "button" in task:
        obj_inds = [5]
    else:
        obj_inds = []
    for obj_ind in obj_inds:
        light_states = [s[obj_ind] for s in scene_states]
        light_states = np.stack(
            [light_states[0]] + light_states, axis=0
        )  # current frame != previous frame
        light_changes = light_states[1:] != light_states[:-1]
        light_changed_inds = np.where(light_changes)[0]
        if light_changed_inds.shape[0] > 0:
            all_changed_inds.extend(light_changed_inds.tolist())

    # For sliding door, drawer, button, and switch, we select the frame
    # before the object is first moved.
    if "slider" in task:
        obj_inds = [0]
    elif "drawer" in task:
        obj_inds = [1]
    elif "led" in task or "button" in task:
        # lightbulb is adjusted by the button
        obj_inds = [2]
    elif "lightbulb" in task or "switch" in task:
        # lightbulb is adjusted by the switch
        obj_inds = [3]
    else:
        obj_inds = []
    for obj_ind in obj_inds:
        object_states = [s[obj_ind] for s in scene_states]
        object_states = np.stack(
            object_states + [object_states[-1]], axis=0
        )  # current frame != future frame
        object_changes = object_states[:-1] != object_states[1:]
        object_changed_inds = np.where(object_changes)[0]
        if object_changed_inds.shape[0] > 0:
            all_changed_inds.append(object_changed_inds.min())

    # For blocks, we subsample the frames where blocks are moved
    if "slider" in task or "drawer" in task or "block" in task:
        object_states = [s[-18:] for s in scene_states]
        object_states = np.stack(
            object_states + [object_states[-1]], axis=0
        )  # current frame != future frame
        object_states = object_states.reshape(-1, 3, 6)
        delta_xyz = np.linalg.norm(
            object_states[:-1, :, :3] - object_states[1:, :, :3], axis=-1
        )
        delta_orn = np.linalg.norm(
            object_states[:-1, :, 3:] - object_states[1:, :, 3:], axis=-1
        )
        object_changes = np.logical_or(delta_xyz > 1e-3, delta_orn > 1e-1)
        object_changed_inds = np.where(object_changes)[0]

        # subsample every 4 frames
        object_changed_inds = object_changed_inds[::6]
        if object_changed_inds.shape[0] > 0:
            all_changed_inds.extend(object_changed_inds.tolist())

    return all_changed_inds


def gripper_state_changed(trajectories):
    trajectories = np.stack(
        [trajectories[0]] + trajectories, axis=0
    )
    openess = trajectories[:, -1]
    changed = openess[:-1] != openess[1:]

    return np.where(changed)[0]


def keypoint_discovery(trajectories, scene_states=None, task=None, buffer_size=5):
    """Determine way point from the trajectories.

    Args:
        trajectories: a list of 1-D np arrays.  Each array is
            7-dimensional (x, y, z, euler_x, euler_y, euler_z, opene).
        stopping_delta: the minimum velocity to determine if the
            end effector is stopped.

    Returns:
        keyframes: list of trajectory points at keyframe indices
        keyframe_inds: array of keyframe indices
        found_natural_maxima: boolean indicating if natural maxima were found
    """
    global stats
    stats['total_sequences'] += 1
    
    print(f"Length of trajectories: {len(trajectories)}")
    
    # Handle empty trajectories case
    if len(trajectories) == 0:
        return [], [], False
        
    V, W, A = get_eef_velocity_from_trajectories(trajectories)
    print(f"Shape of acceleration A: {A.shape}")

    # waypoints are local minima of gripper movement
    _local_max_A = argrelextrema(A, np.greater)[0]
    print(f"Number of local maxima found: {len(_local_max_A)}")
    
    found_natural_maxima = False
    
    # Track original maxima before any fallback
    if len(_local_max_A) > 0:
        stats['sequences_with_maxima'] += 1
        stats['total_maxima_found'] += len(_local_max_A)
        
        # Relaxed constraint: top 50% instead of 20%
        topK = np.sort(A)[::-1][int(A.shape[0] * 0.5)]  # Changed from 0.2 to 0.5
        large_A = A[_local_max_A] >= topK
        _local_max_A = _local_max_A[large_A].tolist()
        print(f"After filtering (top 50%), {len(_local_max_A)} keyframes remain")
        
        if len(_local_max_A) > 0:
            found_natural_maxima = True
    
    # Only proceed with natural maxima or return empty
    if not found_natural_maxima:
        print("No significant natural maxima found, skipping sequence")
        return [], [], False
    
    local_max_A = [_local_max_A[0]]
    for i in _local_max_A[1:]:
        if i - local_max_A[-1] >= buffer_size:
            local_max_A.append(i)

    # waypoints are frames with changing gripper states
    gripper_changed = gripper_state_changed(trajectories)
    one_frame_before_gripper_changed = (
        gripper_changed[gripper_changed > 1] - 1
    )

    # waypoints is the last pose in the trajectory
    last_frame = [len(trajectories) - 1]

    keyframe_inds = (
        local_max_A +
        gripper_changed.tolist() +
        one_frame_before_gripper_changed.tolist() +
        last_frame
    )
    keyframe_inds = np.unique(keyframe_inds)
    print(f"Final keyframes: {keyframe_inds.tolist()}")
    print(f"Final number of keyframes: {len(keyframe_inds)}")

    keyframes = [trajectories[i] for i in keyframe_inds]
    
    if stats['total_sequences'] % 10 == 0:
        print(f"Total sequences processed: {stats['total_sequences']}")
        print(f"Sequences with natural maxima: {stats['sequences_with_maxima']} ({stats['sequences_with_maxima']/stats['total_sequences']*100:.1f}%)")
        print(f"Average maxima found per sequence: {stats['total_maxima_found']/stats['total_sequences']:.2f}")
        print(f"Average maxima used per sequence: {stats['total_maxima_used']/stats['total_sequences']:.2f}\n")

    return keyframes, keyframe_inds, found_natural_maxima


def get_gripper_camera_view_matrix(cam):
    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix


def deproject(cam_params, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a depth image to 3D coordinates using camera parameters
    Args:
        cam_params: dict containing:
            - resolution_width: image width
            - resolution_height: image height
            - fx: focal length x
            - fy: focal length y
            - cx: principal point x
            - cy: principal point y
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output:
        world_pos: (3, npts) or (4, npts) np.array; world coordinates of the deprojected points
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()
    z = depth_img[v, u]

    #convert to 3D coordinates using intrinsic parameters
    x = (u - cam_params['cx']) * z / cam_params['fx']
    y = (v - cam_params['cy']) * z / cam_params['fy']
    z = -z  #negative z because camera looks along negative z aaxis
    
    if homogeneous:
        world_pos = np.stack([x, y, z, np.ones_like(z)], axis=0)
    else:
        world_pos = np.stack([x, y, z], axis=0)

    return world_pos


def convert_rotation(rot):
    """Convert Euler angles to Quarternion
    """
    rot = torch.as_tensor(rot)
    mat = pytorch3d_transforms.euler_angles_to_matrix(rot, "XYZ")
    quat = pytorch3d_transforms.matrix_to_quaternion(mat)
    quat = quat.numpy()

    return quat


def to_relative_action(actions, robot_obs, max_pos=1.0, max_orn=1.0, clip=True):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[..., :3] - robot_obs[..., :3]
    if clip:
        rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos
    else:
        rel_pos = rel_pos / max_pos

    rel_orn = angle_between_angles(robot_obs[..., 3:6], actions[..., 3:6])
    if clip:
        rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn
    else:
        rel_orn = rel_orn / max_orn

    gripper = actions[..., -1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def relative_to_absolute(action, proprio, max_rel_pos=1.0, max_rel_orn=1.0,
                         magic_scaling_factor_pos=1, magic_scaling_factor_orn=1):
    assert action.shape[-1] == 7
    assert proprio.shape[-1] == 7

    rel_pos, rel_orn, gripper = np.split(action, [3, 6], axis=-1)
    rel_pos *= max_rel_pos * magic_scaling_factor_pos
    rel_orn *= max_rel_orn * magic_scaling_factor_orn

    pos_proprio, orn_proprio = proprio[..., :3], proprio[..., 3:6]

    target_pos = pos_proprio + rel_pos
    target_orn = orn_proprio + rel_orn
    return np.concatenate([target_pos, target_orn, gripper], axis=-1)
