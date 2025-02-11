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
We then want to process this data and convert it to a format compatible with 3d-diffuser-actor. Before doing this, we want to produce language instructions to create
correct CLIP embeddings for instructions. To generate npy files for these run:
```bash
$ python3 data_preprocessing/create_lang_annotations.py
```
Then run:
```bash
$ python3 data_preprocessing/package_calvin.py --split training
$ python3 data_preprocessing/package_calvin.py --split validation
```
This should create a new folder called `calvin_processed` with the required structure. We want to then create instruction files in the same format that 3d diffuser actor requires. Run:
```bash
$ python3 data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/training.pkl --model_max_length 16 --annotation_path ./calvin_new/training/lang_annotations/auto_lang_ann.npy
$ python3 data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/validation.pkl --model_max_length 16 --annotation_path ./calvin_new/validation/lang_annotations/auto_lang_ann.npy
```
Now we can train the policy, run:
```bash
$ bash scripts/train_trajectory_calvin.sh
```
