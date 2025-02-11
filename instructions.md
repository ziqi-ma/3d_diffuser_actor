### First create a docker environment:
Note: All of my folders are in `/robo` including 3d-diffuser-actor. Change this path to your folder. 
```bash
$ docker run --gpus all   --runtime=nvidia   -e NVIDIA_DRIVER_CAPABILITIES=all   -e NVIDIA_VISIBLE_DEVICES=all --network=host --privileged     -v $(realpath ~/robo):/workspace -it 3drobo
$ docker start 3drobo
$ docker exec -it 3drobo /bin/bash
```

once in the docker environment, activate calvin conda environment and move to workspace
```bash
$ conda activate calvin
$ cd .. && cd workspace/ 
```
move the run folder to 3d-diffuser-actor folder
Each run folder contains the following:
```
config.json                          rgb_video.mp4                    testing_demo_camera_0_info.npz       testing_demo_ee_states.npz
depth_video.mp4                      testing_demo_action.npz          testing_demo_camera_0_rgba.npz       testing_demo_gripper_states.npz
pov_gripper-2024-12-17_13.32.35.mp4  testing_demo_camera_0_depth.npz  testing_demo_camera_0_timestamp.npz  testing_demo_joint_states.npz
```
```bash
$ mv run10 3d-diffuser-actor/
```
We then want to convert all the raw data to calvin format that can then be preprocessed into 3d-diffuser-actor format.
```bash
$ python3 calvin_converter.py
```
This should create a folder called `calvin_new` in the 3d-diffuser-actor folder.
We then want to process this data and convert it to a format compatible with 3d-diffuser-actor.
```bash
$ python3 data_preprocessing/package_calvin.py --split training
$ python3 data_preprocessing/package_calvin.py --split validation
```
This should create a new folder called `calvin_processed` with the required structure. For now, the indices for training and validation set from calvin data has been hardcoded for `run10` for quick testing. This will be changed soon.
