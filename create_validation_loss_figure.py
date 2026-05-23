"""
Create Validation Loss figure for QLoRA training (1k-4k models)
Generates Figure 8 for the methodology paper
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Validation Loss data from training logs
# X-axis: Normalized epoch values (multiply by 3 for actual epochs 0-3)
epochs_normalized = np.array([0, 0.167, 0.333, 0.5, 0.667, 1.0])
epochs = epochs_normalized * 3  # Convert to actual epochs 0-3

# Validation loss values for each model
val_loss_1k = np.array([3.14, 3.094, 3.081, 3.094, 3.068, 3.135])
val_loss_2k = np.array([3.120, 3.086, 3.073, 3.069, 3.068, 3.073])
val_loss_3k = np.array([3.118, 3.086, 3.074, 3.073, 3.068, 3.073])
val_loss_4k = np.array([3.141, 3.094, 3.081, 3.072, 3.068, 3.067])

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each model with different styles
ax.plot(epochs, val_loss_1k, 'o-', color='red', linewidth=2.5, 
        markersize=8, label='1k samples', alpha=0.8)
ax.plot(epochs, val_loss_2k, 's-', color='blue', linewidth=2.5, 
        markersize=8, label='2k samples', alpha=0.8)
ax.plot(epochs, val_loss_3k, '^--', color='green', linewidth=2.5, 
        markersize=8, label='3k samples', alpha=0.8)
ax.plot(epochs, val_loss_4k, 'D:', color='purple', linewidth=2.5, 
        markersize=8, label='4k samples', alpha=0.8)

# Customize axes
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('PEFT Finetuning Validation Loss', fontsize=16, fontweight='bold', pad=15)

# Set axis limits with some padding
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(3.06, 3.16)

# Grid styling
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend with better positioning
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', 
         fancybox=True, shadow=True)

# Tight layout
plt.tight_layout()

# Save with high DPI for publication
output_path = 'paper_damage_vlm/1_Methodology/figures_vlm/08_validation_loss.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Validation Loss figure saved to: {output_path}")

# Display statistics
print("\n📊 Validation Loss Statistics:")
print(f"1k - Final: {val_loss_1k[-1]:.4f}, Min: {val_loss_1k.min():.4f}")
print(f"2k - Final: {val_loss_2k[-1]:.4f}, Min: {val_loss_2k.min():.4f}")
print(f"3k - Final: {val_loss_3k[-1]:.4f}, Min: {val_loss_3k.min():.4f}")
print(f"4k - Final: {val_loss_4k[-1]:.4f}, Min: {val_loss_4k.min():.4f}")

plt.show()
