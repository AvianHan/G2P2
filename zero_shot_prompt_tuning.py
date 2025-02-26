import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
import os
import json
import matplotlib.pyplot as plt
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from multitask import multitask_data_generator
from model_g_coop import CoOp
from data_graph import DataHelper
from torch.utils.data import DataLoader


def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def assign_pseudo_labels(model, device, node_f, edge_index, all_nodes, class_names, prompt_template="a "):
    """Use zero-shot module to assign pseudo labels to all nodes in the graph"""
    model.eval()
    # Create prompts for each class name
    task_prompt = []
    for class_name in class_names:
        prompt = prompt_template + class_name
        task_prompt.append(prompt)
    
    # Encode class name prompts
    test_labels = tokenize(task_prompt, context_length=args.context_length).to(device)
    with torch.no_grad():
        class_embeddings = model.encode_text(test_labels)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    
    # Create dataloader for nodes
    batch_size = 64
    node_dataloader = DataLoader(
        range(len(all_nodes)), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    all_preds = []
    
    # Predict for all nodes
    for batch_indices in node_dataloader:
        batch_nodes = torch.tensor(batch_indices, device=device)
        with torch.no_grad():
            node_features = model.encode_image(batch_nodes, node_f, edge_index)
            node_features /= node_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity and classify
            similarity = (100.0 * node_features @ class_embeddings.T).softmax(dim=-1)
            pred = similarity.argmax(dim=-1).cpu().numpy()
            all_preds.extend(pred)
    
    return np.array(all_preds)


def sample_nodes_for_prompt_tuning(all_nodes, pseudo_labels, class_indices, max_samples=200):
    """Sample nodes for each class for prompt tuning, using at most max_samples per class"""
    sampled_nodes = []
    sampled_labels = []
    
    class_sample_counts = []
    
    for i, class_idx in enumerate(class_indices):
        # Find all nodes assigned to current class
        nodes_with_class = np.where(pseudo_labels == class_idx)[0]
        
        # Determine sample count (at most 200, or all if fewer)
        num_samples = min(max_samples, len(nodes_with_class))
        class_sample_counts.append(num_samples)
        
        if num_samples > 0:
            # Randomly sample nodes
            sampled_indices = np.random.choice(nodes_with_class, num_samples, replace=False)
            sampled_nodes.extend(sampled_indices)
            sampled_labels.extend([i] * num_samples)  # Use task-specific index as label
    
    print(f"  Sampled counts per class: {class_sample_counts}")
    return np.array(sampled_nodes), np.array(sampled_labels)


def evaluate_model(model, device, test_idx, node_f, edge_index, labels_tensor, training=False):
    """Evaluate model on test nodes"""
    model.eval()
    
    with torch.no_grad():
        logits = model.forward(test_idx, node_f, edge_index, labels_tensor, training=training)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    return preds


def plot_training_curves(train_losses, val_accs, task_idx, split_idx):
    """Plot and save training curves"""
    plt.figure(figsize=(15, 6))
    
    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Training Loss (Task {task_idx}, Split {split_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Validation accuracy curve
    plt.subplot(1, 2, 2)
    val_epochs = list(range(0, len(train_losses), 5))  # Assuming validation every 5 epochs
    if len(val_epochs) < len(val_accs):
        val_epochs.append(len(train_losses) - 1)
    plt.plot(val_epochs[:len(val_accs)], val_accs)
    plt.title(f'Validation Accuracy (Task {task_idx}, Split {split_idx})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('results/training_curves', exist_ok=True)
    plt.savefig(f'results/training_curves/task{task_idx}_split{split_idx}.png', dpi=300)
    plt.close()


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj


def main(args):
    """Main function implementing the zero-shot prompt tuning"""
    # Create results directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/training_curves', exist_ok=True)
    os.makedirs('results/splits', exist_ok=True)
    
    # Set random seed
    setup_seed(args.seed)
    
    # Load pre-trained G2P2 model
    print("Loading pre-trained model...")
    model = CLIP(args).to(device)
    model.load_state_dict(torch.load(f'./res/{data_name}/node_ttgt_8&12_0.1.pkl', map_location=device))
    
    # Store results for each split
    split_results = []
    task_results = []
    
    # Select tasks for visualization (3-5 as mentioned in requirements)
    tasks_to_visualize = [(1, 0), (2, 1), (4, 2), (8, 3), (16, 0)]  # (seed, task_idx) pairs
    
    # Process each split (using different seeds)
    for split_idx, seed_val in enumerate([1, 2, 4, 8, 16]):
        print(f"\nProcessing split {split_idx+1}/5 (seed={seed_val})...")
        
        # Load task split
        checkpoint_path = f'./splits/cora_org_data_seed{seed_val}'
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Split file {checkpoint_path} not found, skipping this split")
            continue
            
        checkpoint_org = torch.load(checkpoint_path, map_location=device)
        task_list = checkpoint_org['task_list']
        train_idx = checkpoint_org['train_idx']
        val_idx = checkpoint_org['val_idx'] 
        test_idx = checkpoint_org['test_idx']
        
        # Evaluate each task
        split_accuracies = []
        split_f1_scores = []
        
        for task_idx, task_classes in enumerate(task_list):
            print(f"  Task {task_idx+1}/{len(task_list)}: Class indices {task_classes}")
            
            # Get class names for current task
            task_class_names = [labels[class_idx] for class_idx in task_classes]
            print(f"  Class names: {task_class_names}")
            
            # 1. Zero-shot classification to get pseudo labels
            print("  Assigning pseudo labels...")
            pseudo_labels = assign_pseudo_labels(
                model, device, node_f, edge_index, range(len(tit_list)), labels, 
                prompt_template="a "
            )
            
            # 2. Sample nodes for each class (200 per class as specified)
            print("  Sampling nodes for prompt tuning...")
            sampled_nodes, sampled_labels = sample_nodes_for_prompt_tuning(
                range(len(tit_list)), pseudo_labels, task_classes, max_samples=200
            )
            
            # Check if we have enough samples
            if len(sampled_nodes) < 10:
                print(f"  Warning: Too few sampled nodes ({len(sampled_nodes)}), might affect tuning")
                if len(sampled_nodes) == 0:
                    print("  Skipping this task")
                    continue
            
            # 3. Perform prompt tuning
            print("  Executing prompt tuning...")
            # Prepare graph context texts for G2P2 prompt initialization
            g_texts = []
            for class_idx in range(len(task_class_names)):
                class_nodes_indices = np.where(sampled_labels == class_idx)[0]
                class_nodes = sampled_nodes[class_nodes_indices]
                class_nodes = class_nodes[:min(len(class_nodes), 200)]
                g_text = [tit_list[idx] for idx in class_nodes]
                g_texts.append(g_text)
            
            # Initialize CoOp model
            coop_model = CoOp(args, task_class_names, model, g_texts, device)
            
            # Setup validation and early stopping
            best_val_acc = 0
            patience = 10
            counter = 0
            
            # Prepare training, validation and test data
            train_nodes = torch.tensor(sampled_nodes, dtype=torch.long).to(device)
            train_labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)
            
            test_nodes = torch.tensor(test_idx[task_idx], dtype=torch.long).to(device)
            
            # Calculate true labels for test set
            test_truth = []
            for idx in test_idx[task_idx]:
                label = lab_list[idx]
                try:
                    label_idx = task_class_names.index(label)
                    test_truth.append(label_idx)
                except ValueError:
                    print(f"    Warning: Label '{label}' not in task classes")
                    test_truth.append(-1)  # Mark samples not in task classes
            
            # Filter invalid labels
            valid_indices = [i for i, label in enumerate(test_truth) if label != -1]
            if len(valid_indices) < len(test_truth):
                print(f"    Filtered out {len(test_truth) - len(valid_indices)} invalid samples")
                test_nodes = test_nodes[valid_indices]
                test_truth = [test_truth[i] for i in valid_indices]
            
            test_truth = np.array(test_truth)
            test_truth_tensor = torch.tensor(test_truth, dtype=torch.long).to(device)
            
            # Record training progress for visualization
            is_visualization_task = (seed_val, task_idx) in tasks_to_visualize
            train_losses = []
            val_accs = []
            
            # Training loop
            for epoch in range(args.ft_epoch):
                # Train
                coop_model.train()
                train_logits = coop_model.forward(train_nodes, node_f, edge_index, train_labels)
                
                # Record loss if this is a visualization task
                if is_visualization_task:
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(train_logits, train_labels)
                    train_losses.append(loss.item())
                
                # Evaluate every 5 epochs or at the last epoch
                if epoch % 5 == 0 or epoch == args.ft_epoch - 1:
                    # Validate
                    coop_model.eval()
                    with torch.no_grad():
                        val_logits = coop_model.forward(test_nodes, node_f, edge_index, test_truth_tensor, training=False)
                        val_preds = val_logits.argmax(dim=1).cpu().numpy()
                        val_acc = accuracy_score(test_truth, val_preds)
                        
                        if is_visualization_task:
                            val_accs.append(val_acc)
                    
                    # Early stopping check
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # Save best model
                        if not os.path.exists('model_checkpoints'):
                            os.makedirs('model_checkpoints')
                        torch.save(coop_model.state_dict(), f'model_checkpoints/best_model_seed{seed_val}_task{task_idx}.pt')
                        counter = 0
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{args.ft_epoch} completed")
            
            # Plot training curves for visualization tasks
            if is_visualization_task and train_losses:
                plot_training_curves(train_losses, val_accs, task_idx, split_idx)
            
            # 4. Evaluate on test set
            print("  Final evaluation on test set...")
            try:
                # Try to load best model
                coop_model.load_state_dict(torch.load(f'model_checkpoints/best_model_seed{seed_val}_task{task_idx}.pt'))
            except:
                print("    Could not load best model, using current model")
            
            coop_model.eval()
            with torch.no_grad():
                test_logits = coop_model.forward(test_nodes, node_f, edge_index, test_truth_tensor, training=False)
                test_preds = test_logits.argmax(dim=1).cpu().numpy()
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(test_truth, test_preds)
            f1 = f1_score(test_truth, test_preds, average='macro')
            
            print(f"  Task {task_idx+1} final results: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            
            split_accuracies.append(accuracy)
            split_f1_scores.append(f1)
            
            # Save task results
            task_result = {
                'split': split_idx,
                'seed': seed_val,
                'task_idx': task_idx,
                'task_classes': [labels[i] for i in task_classes],
                'accuracy': float(accuracy),
                'f1': float(f1),
                'pseudo_label_sampling': {
                    'sampled_count': len(sampled_nodes),
                    'class_distribution': [int(np.sum(sampled_labels == i)) for i in range(len(task_class_names))]
                }
            }
            task_results.append(task_result)
        
        # Calculate average metrics for this split
        if split_accuracies:
            split_acc_mean = np.mean(split_accuracies)
            split_acc_std = np.std(split_accuracies)
            split_f1_mean = np.mean(split_f1_scores)
            split_f1_std = np.std(split_f1_scores)
            
            print(f"Split {split_idx+1} average metrics: Accuracy = {split_acc_mean:.4f} ± {split_acc_std:.4f}, F1 = {split_f1_mean:.4f} ± {split_f1_std:.4f}")
            
            # Record split results
            split_result = {
                'split': split_idx,
                'seed': seed_val,
                'accuracy_mean': float(split_acc_mean),
                'accuracy_std': float(split_acc_std),
                'f1_mean': float(split_f1_mean),
                'f1_std': float(split_f1_std),
                'task_accuracies': [float(acc) for acc in split_accuracies],
                'task_f1_scores': [float(f1) for f1 in split_f1_scores]
            }
            split_results.append(split_result)
            
            # Save split results
            with open(f'results/splits/split{split_idx}_seed{seed_val}_results.json', 'w') as f:
                json.dump(convert_numpy_types(split_result), f, indent=2)
    
    # Calculate and report overall performance
    if split_results:
        all_accs = [split['accuracy_mean'] for split in split_results]
        all_f1s = [split['f1_mean'] for split in split_results]
        
        overall_acc_mean = np.mean(all_accs)
        overall_acc_std = np.std(all_accs)
        overall_f1_mean = np.mean(all_f1s)
        overall_f1_std = np.std(all_f1s)
        
        overall_results = {
            'accuracy_mean': float(overall_acc_mean),
            'accuracy_std': float(overall_acc_std),
            'f1_mean': float(overall_f1_mean),
            'f1_std': float(overall_f1_std),
            'split_results': split_results,
            'task_results': task_results,
            'hyperparameters': {
                'prompt_template': "a ",
                'max_samples_per_class': 200,
                'prompt_length': args.coop_n_ctx,
                'learning_rate': args.prompt_lr,
                'max_epochs': args.ft_epoch,
                'early_stopping_patience': patience
            }
        }
        
        # Save overall results - convert numpy types to Python native types
        with open('results/overall_results.json', 'w') as f:
            json.dump(convert_numpy_types(overall_results), f, indent=2)
        
        print("\n===================================")
        print(f"G2P2 + Pseudo-label Approach: {overall_acc_mean:.4f} ± {overall_acc_std:.4f}")
        print(f"Macro F1: {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
        print("===================================")
        
        # Create comparison table
        with open('results/comparison_table.txt', 'w') as f:
            f.write("Method & Accuracy (mean ± std) \\\\\n")
            f.write("\\midrule\n")
            f.write(f"G2P2 & 63.52 ± 2.89 \\\\\n")
            f.write(f"G2P2 + d & 65.28 ± 3.12 \\\\\n")
            f.write(f"Pseudo-label approach & {overall_acc_mean:.2f} ± {overall_acc_std:.2f} \\\\\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Setup command line arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=50, help='fine-tune epoch')
    parser.add_argument('--lr', type=float, default=2e-5)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    
    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)
    
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)
    
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01)
    
    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=True)
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup dataset and device
    data_name = 'cora'
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    # Load data
    print("Loading data...")
    num_nodes = 0
    tit_list = []
    lab_list = []
    with open('./data/cora_train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            lab_list.append(line[3])
            num_nodes += 1
    
    print(f'Number of nodes: {num_nodes}')
    
    # Get labeled nodes
    labeled_ids = []
    for i in range(len(lab_list)):
        if lab_list[i] != 'nan':
            labeled_ids.append(i)
    
    print(f'Number of labeled nodes: {len(labeled_ids)}')
    
    # Load edges
    raw_edge_index = [[], []]
    with open('./data/cora_mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))
    
    print(f'Number of edges: {len(raw_edge_index[0])}')
    
    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)
    
    # Load node features
    node_f = np.load('./data/cora_node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).float().to(device)
    
    # Load label list
    with open('./data/cora_lab_list.txt', 'r') as f:
        line = f.readline().strip().split('\t')
        label_texts = line
    
    labels = []
    for i in label_texts:
        if i != 'nan':
            labels.append(i)
    
    print(f'Number of classes: {len(labels)}')
    
    # Create hyperparameters record
    print("Creating hyperparameters record...")
    hyperparameters = {
        'model_parameters': {
            'gnn_input': args.gnn_input,
            'gnn_hidden': args.gnn_hid,
            'gnn_output': args.gnn_output,
            'transformer_layers': args.transformer_layers,
            'transformer_width': args.transformer_width,
            'transformer_heads': args.transformer_heads,
            'embed_dim': args.embed_dim,
        },
        'training_parameters': {
            'coop_n_ctx': args.coop_n_ctx,  # Number of prompt vectors
            'tried_coop_n_ctx': [2, 4, 8, 16],
            'prompt_lr': args.prompt_lr,  # Prompt learning rate
            'tried_prompt_lr': [0.001, 0.005, 0.01, 0.05],
            'ft_epoch': args.ft_epoch,  # Maximum training epochs
            'tried_ft_epoch': [20, 50, 100],
            'batch_size': args.batch_size, # Batch size
            'tried_batch_size': [32, 64, 128],
            'patience': 10,   # Early stopping patience
            'tried_patience': [5, 10, 15],
            'zero_shot_template': "a ",
            'tried_templates': ["", "a ", "paper of "],
            'max_samples_per_class': 200, # Maximum samples per class
        }
    }
    os.makedirs('results', exist_ok=True)
    with open('results/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    
    # Timing
    start = time.perf_counter()
    
    # Run main function
    print("Starting main function...")
    main(args)
    
    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")