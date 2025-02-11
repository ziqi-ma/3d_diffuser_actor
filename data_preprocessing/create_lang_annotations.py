import numpy as np
import os
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
import torch

def generate_language_annotations(training_dir, val_dir, task_description="pick the mug and place it down"):
    """Generate language annotations file for CALVIN dataset format."""
    
    # Get episode files and sort them
    train_episodes = sorted([f for f in os.listdir(training_dir) if f.endswith('.npz')])
    val_episodes = sorted([f for f in os.listdir(val_dir) if f.endswith('.npz')])
    
    # Extract start and end indices from filenames
    train_indices = [int(f.split('_')[1].split('.')[0]) for f in train_episodes]
    val_indices = [int(f.split('_')[1].split('.')[0]) for f in val_episodes]
    
    # Create separate annotations for training and validation
    def create_split_annotations(episodes, start_idx, end_idx):
        # For a single task, we'll create one sequence that encompasses all episodes
        indices = [(start_idx, end_idx)]
        
        # Initialize CLIP model for embeddings (same as CALVIN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # Generate embedding for the task description
        with torch.no_grad():
            tokens = tokenizer(task_description, padding="max_length", return_tensors="pt").to(device)
            embedding = model(**tokens).last_hidden_state.cpu().numpy()
        
        # Create the annotation structure
        lang_ann = {
            'language': {
                'ann': [task_description],    # Single task description
                'task': ['pick_mug'],         # Single task identifier
                'emb': embedding              # Shape will be (1, 1, 384)
            },
            'info': {
                'episodes': [],               # Empty list as per original structure
                'indx': indices              # Single tuple with start and end indices
            }
        }
        
        return lang_ann
    
    # Generate annotations for training split
    train_ann = create_split_annotations(
        train_episodes,
        min(train_indices),
        max(train_indices)
    )
    
    # Generate annotations for validation split
    val_ann = create_split_annotations(
        val_episodes,
        min(val_indices),
        max(val_indices)
    )
    
    # Save the annotations
    os.makedirs(os.path.join(training_dir, "lang_annotations"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "lang_annotations"), exist_ok=True)
    
    np.save(os.path.join(training_dir, "lang_annotations/auto_lang_ann.npy"), train_ann)
    np.save(os.path.join(val_dir, "lang_annotations/auto_lang_ann.npy"), val_ann)
    
    return train_ann, val_ann

# Example usage
if __name__ == "__main__":
    training_dir = "/workspace/3d_diffuser_actor/calvin_new/training"
    val_dir = "/workspace/3d_diffuser_actor/calvin_new/validation"
    
    train_ann, val_ann = generate_language_annotations(training_dir, val_dir)
    
    # Print structure to verify
    print("\nGenerated training annotation structure:")
    print(f"Number of annotations: {len(train_ann['language']['ann'])}")
    print(f"Embedding shape: {train_ann['language']['emb'].shape}")
    print(f"Indices: {train_ann['info']['indx']}")
    
    print("\nGenerated validation annotation structure:")
    print(f"Number of annotations: {len(val_ann['language']['ann'])}")
    print(f"Embedding shape: {val_ann['language']['emb'].shape}")
    print(f"Indices: {val_ann['info']['indx']}")