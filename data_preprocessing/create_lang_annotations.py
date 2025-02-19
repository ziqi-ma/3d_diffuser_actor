import numpy as np
import os
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
import torch
import json

def generate_language_annotations(training_dir, val_dir, task_description="pick the cube and place it down"):
    """Generate language annotations file for CALVIN dataset format."""
    
    # Load run info to get proper episode ranges
    run_info_file = Path(training_dir).parent / 'run_info.json'
    with open(run_info_file, 'r') as f:
        runs = json.load(f)
    
    # Create sequences from run info
    train_sequences = []
    val_sequences = []
    
    for run in runs:
        start_idx = run['start_idx']
        end_idx = run['end_idx']
        n_train = run['n_train']
        
        # Add training sequence
        if n_train > 0:
            train_sequences.append((start_idx, start_idx + n_train - 1))
        
        # Add validation sequence
        if end_idx > (start_idx + n_train - 1):
            val_sequences.append((start_idx + n_train, end_idx))
    
    # Initialize CLIP model for embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Generate embedding for the task description
    with torch.no_grad():
        tokens = tokenizer(task_description, padding="max_length", return_tensors="pt").to(device)
        embedding = model(**tokens).last_hidden_state.cpu().numpy()
    
    # Create annotations for training
    train_ann = {
        'language': {
            'ann': [task_description] * len(train_sequences),
            'task': ['pick_cube'] * len(train_sequences),
            'emb': np.repeat(embedding, len(train_sequences), axis=0)
        },
        'info': {
            'episodes': [],
            'indx': train_sequences
        }
    }
    
    # Create annotations for validation
    val_ann = {
        'language': {
            'ann': [task_description] * len(val_sequences),
            'task': ['pick_cube'] * len(val_sequences),
            'emb': np.repeat(embedding, len(val_sequences), axis=0)
        },
        'info': {
            'episodes': [],
            'indx': val_sequences
        }
    }
    
    # Save the annotations
    os.makedirs(os.path.join(training_dir, "lang_annotations"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "lang_annotations"), exist_ok=True)
    
    np.save(os.path.join(training_dir, "lang_annotations/auto_lang_ann.npy"), train_ann)
    np.save(os.path.join(val_dir, "lang_annotations/auto_lang_ann.npy"), val_ann)
    
    return train_ann, val_ann

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, default="/workspace/3d_diffuser_actor/calvin_complete/training")
    parser.add_argument('--val_dir', type=str, default="/workspace/3d_diffuser_actor/calvin_complete/validation")
    args = parser.parse_args()
    
    train_ann, val_ann = generate_language_annotations(args.training_dir, args.val_dir)
    
    # Print structure to verify
    print("\nGenerated training annotation structure:")
    print(f"Number of annotations: {len(train_ann['language']['ann'])}")
    print(f"Embedding shape: {train_ann['language']['emb'].shape}")
    print(f"Indices: {train_ann['info']['indx']}")
    
    print("\nGenerated validation annotation structure:")
    print(f"Number of annotations: {len(val_ann['language']['ann'])}")
    print(f"Embedding shape: {val_ann['language']['emb'].shape}")
    print(f"Indices: {val_ann['info']['indx']}")