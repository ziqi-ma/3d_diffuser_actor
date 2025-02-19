#!/bin/bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29501 \
    main_trajectory_calvin.py \
    --exp_log_dir calvin_task_ABC_D_complete \
    --run_log_dir run0 \
    --instructions instructions/calvin_task_ABC_D_complete/training.pkl \
    --val_instructions instructions/calvin_task_ABC_D_complete/validation.pkl \
    --training_dir calvin_complete_processed/training \
    --val_dir calvin_complete_processed/validation \
    --batch_size 32 \
    --num_workers 8 \
    --max_episode_length 5 \
    --dense_interpolation \
    --interpolation_length 20 \
    --num_epochs 100
