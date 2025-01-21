main_dir=Planner_Calvin

dataset=/data/calvin/calvin_debug_dataset/validation
valset=/data/calvin/calvin_debug_dataset/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=3
diffusion_timesteps=25
B=30
C=192
ngpus=6
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint

export PYTHONPATH=`pwd`:$PYTHONPATH

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path /data/calvin/calvin_debug_dataset \
    --calvin_model_path /app/calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds /data/calvin/calvin_debug_dataset/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 0 \
    --base_log_dir train_logs/${main_dir}/pretrained/eval_logs/ \
    --quaternion_format $quaternion_format \
    --checkpoint /data/checkpoints/diffuseractor/diffuser_actor_calvin.pth
