import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory where k-fold results are saved
RESULTS_DIR = 'kfold_results'
OUTPUT_DIR = 'kfold_visualizations'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to visualize
models = ['efficientnet_b0', 'resnet18', 'densenet121', 'vgg16']
model_display_names = {
    'efficientnet_b0': 'EfficientNet-B0',
    'resnet18': 'ResNet18',
    'densenet121': 'DenseNet121',
    'vgg16': 'VGG16'
}

def load_kfold_results(model_name):
    """Load k-fold results from CSV file."""
    csv_path = os.path.join(RESULTS_DIR, f"{model_name}_kfold_results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)

def plot_fold_comparison():
    """Bar chart comparing accuracy across folds for all models."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(5)  # 5 folds
    width = 0.2
    
    for i, model in enumerate(models):
        df = load_kfold_results(model)
        if df is not None:
            accs = df['best_val_acc'].values
            offset = (i - 1.5) * width
            ax.bar(x + offset, accs, width, 
                   label=model_display_names[model], alpha=0.8)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation: Accuracy per Fold', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.97, 1.005])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kfold_comparison_by_fold.png'), dpi=300)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/kfold_comparison_by_fold.png")

def plot_mean_std_comparison():
    """Bar chart with error bars showing mean and std dev."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = []
    stds = []
    labels = []
    
    for model in models:
        df = load_kfold_results(model)
        if df is not None:
            accs = df['best_val_acc'].values
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            labels.append(model_display_names[model])
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.8, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation: Mean Accuracy with Standard Deviation', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.98, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kfold_mean_std_comparison.png'), dpi=300)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/kfold_mean_std_comparison.png")

def plot_box_plot():
    """Box plot showing distribution of accuracies across folds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    
    for model in models:
        df = load_kfold_results(model)
        if df is not None:
            data.append(df['best_val_acc'].values)
            labels.append(model_display_names[model])
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation: Accuracy Distribution', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.97, 1.005])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kfold_boxplot.png'), dpi=300)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/kfold_boxplot.png")

def plot_training_time_comparison():
    """Bar chart comparing training time per fold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_times = []
    labels = []
    
    for model in models:
        df = load_kfold_results(model)
        if df is not None:
            times = df['time'].values / 60  # Convert to minutes
            mean_times.append(np.mean(times))
            labels.append(model_display_names[model])
    
    x = np.arange(len(labels))
    bars = ax.bar(x, mean_times, alpha=0.8, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Training Time per Fold (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation: Training Time Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, mean_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f} min',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kfold_training_time.png'), dpi=300)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/kfold_training_time.png")

def plot_individual_model_details(model_name):
    """Detailed plot for a single model showing train vs val accuracy."""
    df = load_kfold_results(model_name)
    if df is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Train vs Val Accuracy per fold
    folds = df['fold'].values
    train_accs = df['final_train_acc'].values
    val_accs = df['final_val_acc'].values
    
    x = np.arange(len(folds))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Train Acc', alpha=0.8)
    ax1.bar(x + width/2, val_accs, width, label='Val Acc', alpha=0.8)
    ax1.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title(f'{model_display_names[model_name]}: Train vs Val Accuracy', 
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {f}' for f in folds])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.97, 1.005])
    
    # Plot 2: Best Val Accuracy per fold
    best_val_accs = df['best_val_acc'].values
    ax2.plot(folds, best_val_accs, marker='o', markersize=10, linewidth=2)
    ax2.axhline(y=np.mean(best_val_accs), color='r', linestyle='--', 
                label=f'Mean: {np.mean(best_val_accs):.4f}')
    ax2.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Best Validation Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title(f'{model_display_names[model_name]}: Best Val Accuracy per Fold', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.97, 1.005])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_detailed.png'), dpi=300)
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/{model_name}_detailed.png")

def create_summary_table():
    """Create and save a summary table of all results."""
    summary_data = []
    
    for model in models:
        df = load_kfold_results(model)
        if df is not None:
            accs = df['best_val_acc'].values
            times = df['time'].values / 60  # Convert to minutes
            summary_data.append({
                'Model': model_display_names[model],
                'Mean Acc': f"{np.mean(accs):.4f}",
                'Std Acc': f"{np.std(accs):.4f}",
                'Min Acc': f"{np.min(accs):.4f}",
                'Max Acc': f"{np.max(accs):.4f}",
                'Avg Time (min)': f"{np.mean(times):.1f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_path = os.path.join(OUTPUT_DIR, 'kfold_summary_table.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary table: {summary_path}")
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VISUALIZING K-FOLD CROSS-VALIDATION RESULTS")
    print("="*60 + "\n")
    
    # Generate all visualizations
    print("1. Creating fold-by-fold comparison...")
    plot_fold_comparison()
    
    print("\n2. Creating mean accuracy with error bars...")
    plot_mean_std_comparison()
    
    print("\n3. Creating box plot...")
    plot_box_plot()
    
    print("\n4. Creating training time comparison...")
    plot_training_time_comparison()
    
    print("\n5. Creating detailed plots for each model...")
    for model in models:
        plot_individual_model_details(model)
    
    print("\n6. Creating summary table...")
    create_summary_table()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("="*60)
