"""
Visualize ground truth and prediction heatmaps from prediction results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def load_grid_dimensions(city: str) -> tuple:
    """
    Get grid dimensions for the city.
    Beijing: 40x40, Shenzhen: 30x30, Nanchang: 20x20
    """
    grid_dims = {
        'beijing': (40, 40),
        'shenzhen': (30, 30),
        'nanchang': (20, 20)
    }
    return grid_dims.get(city.lower(), (40, 40))


def grid_id_to_coords(grid_id: int, grid_size: int) -> tuple:
    """Convert grid ID to (row, col) coordinates."""
    row = grid_id // grid_size
    col = grid_id % grid_size
    return row, col


def create_heatmap_from_locations(location_ids: list, grid_rows: int, grid_cols: int, 
                                  normalize: bool = True) -> np.ndarray:
    """
    Create a heatmap matrix from location IDs.
    
    Args:
        location_ids: List of location IDs
        grid_rows: Number of grid rows
        grid_cols: Number of grid columns
        normalize: If True, normalize to [0, 1]
    
    Returns:
        2D numpy array representing the heatmap
    """
    heatmap = np.zeros((grid_rows, grid_cols))
    
    for loc_id in location_ids:
        if loc_id >= 0:  # Ignore invalid locations (-1)
            row, col = grid_id_to_coords(loc_id, grid_cols)
            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                heatmap[row, col] += 1
    
    if normalize and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def visualize_ground_truth_heatmap(csv_file: str, output_file: str = None, 
                                   city: str = 'beijing', dpi: int = 300):
    """
    Visualize ground truth locations as a heatmap.
    
    Args:
        csv_file: Path to predictions CSV file
        output_file: Output image file path
        city: City name for grid dimensions
        dpi: DPI for output image
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    print(f"Grid dimensions: {grid_rows}x{grid_cols}")
    
    # Extract ground truth locations
    ground_truth = df['ground_truth'].values
    
    # Create heatmap
    print("Creating heatmap...")
    heatmap = create_heatmap_from_locations(ground_truth, grid_rows, grid_cols, normalize=False)
    
    # Statistics
    unique_locations = len(np.unique(ground_truth[ground_truth >= 0]))
    total_visits = len(ground_truth)
    max_visits = int(heatmap.max())
    
    print(f"\nStatistics:")
    print(f"  Total predictions: {total_visits}")
    print(f"  Unique locations: {unique_locations}")
    print(f"  Most visited location: {max_visits} times")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Visits', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)
    ax.set_title(f'Ground Truth Location Heatmap - {city.title()}\n'
                f'Total Predictions: {total_visits} | Unique Locations: {unique_locations}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
    ax.set_xticks(np.arange(0, grid_cols, 5))
    ax.set_yticks(np.arange(0, grid_rows, 5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_file:
        print(f"\nSaving heatmap to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction_vs_ground_truth(csv_file: str, output_file: str = None,
                                        city: str = 'beijing', dpi: int = 300):
    """
    Visualize ground truth vs top-1 predictions side by side.
    
    Args:
        csv_file: Path to predictions CSV file
        output_file: Output image file path
        city: City name for grid dimensions
        dpi: DPI for output image
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    
    # Extract locations
    ground_truth = df['ground_truth'].values
    predictions = df['top1_prediction'].values
    
    # Create heatmaps
    print("Creating heatmaps...")
    gt_heatmap = create_heatmap_from_locations(ground_truth, grid_rows, grid_cols, normalize=False)
    pred_heatmap = create_heatmap_from_locations(predictions, grid_rows, grid_cols, normalize=False)
    
    # Calculate accuracy
    accuracy = df['top1_correct'].mean() * 100
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Common colormap limits
    vmax = max(gt_heatmap.max(), pred_heatmap.max())
    
    # Plot ground truth
    im1 = axes[0].imshow(gt_heatmap, cmap='YlOrRd', aspect='auto', origin='lower', vmax=vmax)
    axes[0].set_xlabel('Grid Column', fontsize=11)
    axes[0].set_ylabel('Grid Row', fontsize=11)
    axes[0].set_title('Ground Truth Locations', fontsize=13, fontweight='bold')
    axes[0].grid(True, which='both', alpha=0.2, linewidth=0.5)
    axes[0].set_xticks(np.arange(0, grid_cols, 5))
    axes[0].set_yticks(np.arange(0, grid_rows, 5))
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Visits', rotation=270, labelpad=15, fontsize=10)
    
    # Plot predictions
    im2 = axes[1].imshow(pred_heatmap, cmap='YlOrRd', aspect='auto', origin='lower', vmax=vmax)
    axes[1].set_xlabel('Grid Column', fontsize=11)
    axes[1].set_ylabel('Grid Row', fontsize=11)
    axes[1].set_title('Top-1 Predictions', fontsize=13, fontweight='bold')
    axes[1].grid(True, which='both', alpha=0.2, linewidth=0.5)
    axes[1].set_xticks(np.arange(0, grid_cols, 5))
    axes[1].set_yticks(np.arange(0, grid_rows, 5))
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Visits', rotation=270, labelpad=15, fontsize=10)
    
    # Overall title
    fig.suptitle(f'Ground Truth vs Predictions - {city.title()}\n'
                f'Samples: {len(df)} | Top-1 Accuracy: {accuracy:.2f}%',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show
    if output_file:
        print(f"\nSaving comparison to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def visualize_top10_distribution(csv_file: str, output_file: str = None,
                                 city: str = 'beijing', dpi: int = 300):
    """
    Visualize distribution of all top-10 predictions weighted by confidence.
    
    Args:
        csv_file: Path to predictions CSV file
        output_file: Output image file path
        city: City name for grid dimensions
        dpi: DPI for output image
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    
    # Create weighted heatmap
    heatmap = np.zeros((grid_rows, grid_cols))
    
    print("Creating weighted heatmap from top-10 predictions...")
    for _, row in df.iterrows():
        for i in range(1, 11):  # Top 10 predictions
            pred_col = f'pred_{i}'
            conf_col = f'conf_{i}'
            
            if pred_col in df.columns and conf_col in df.columns:
                pred_id = row[pred_col]
                confidence = row[conf_col]
                
                if pred_id >= 0 and confidence > 0:
                    r, c = grid_id_to_coords(int(pred_id), grid_cols)
                    if 0 <= r < grid_rows and 0 <= c < grid_cols:
                        heatmap[r, c] += confidence
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(heatmap, cmap='viridis', aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weighted Confidence Sum', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)
    ax.set_title(f'Top-10 Predictions Distribution (Confidence-Weighted) - {city.title()}\n'
                f'Total Samples: {len(df)}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
    ax.set_xticks(np.arange(0, grid_cols, 5))
    ax.set_yticks(np.arange(0, grid_rows, 5))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        print(f"\nSaving weighted distribution to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def visualize_raw_data_heatmap(city: str, data_type: str = 'val', 
                               output_file: str = None, dpi: int = 300):
    """
    Visualize location distribution from raw data files (train.csv, val.csv, test.csv).
    This allows checking the data distribution before making predictions.
    
    Args:
        city: City name (beijing, shenzhen, nanchang)
        data_type: Type of data file (train, val, test)
        output_file: Output image file path
        dpi: DPI for output image
    """
    # Construct data file path
    # data_file = f'/workspace/China_Journal/data/{city}/{data_type}.csv'
    data_file = '/workspace/China_Journal/visual/test_for_visual_beijing.csv'
    
    # Load data
    print(f"Loading raw data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found!")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    print(f"Grid dimensions: {grid_rows}x{grid_cols}")
    
    # Extract location IDs
    location_ids = df['location_id'].values
    
    # Create heatmap
    print("Creating heatmap...")
    heatmap = create_heatmap_from_locations(location_ids, grid_rows, grid_cols, normalize=False)
    
    # Statistics
    unique_locations = len(np.unique(location_ids[location_ids >= 0]))
    total_records = len(location_ids)
    max_visits = int(heatmap.max())
    unique_users = df['user_id'].nunique() if 'user_id' in df.columns else 'N/A'
    
    print(f"\nStatistics:")
    print(f"  Total records: {total_records}")
    print(f"  Unique users: {unique_users}")
    print(f"  Unique locations: {unique_locations}")
    print(f"  Most visited location: {max_visits} times")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Plot heatmap
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Visits', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)
    ax.set_title(f'{data_type.upper()} Data Location Heatmap - {city.title()}\n'
                f'Records: {total_records:,} | Users: {unique_users} | Unique Locations: {unique_locations}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
    ax.set_xticks(np.arange(0, grid_cols, 5))
    ax.set_yticks(np.arange(0, grid_rows, 5))
    
    # Add statistics text box
    stats_text = f'Max visits: {max_visits}\nSparsity: {(heatmap > 0).sum()}/{grid_rows*grid_cols} grids'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_file:
        print(f"\nSaving heatmap to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize prediction results as heatmaps')
    parser.add_argument('--csv_file', type=str, 
                       default='/workspace/China_Journal/visual/predictions_beijing.csv',
                       help='Path to predictions CSV file (for prediction modes)')
    parser.add_argument('--city', type=str, default='beijing',
                       choices=['beijing', 'shenzhen', 'nanchang'],
                       help='City name for grid dimensions')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for saved images. Default: same as CSV file or visual/')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images')
    parser.add_argument('--mode', type=str, default='raw_data',
                       choices=['ground_truth', 'comparison', 'top10', 'all', 'raw_data'],
                       help='Visualization mode. Use "raw_data" to visualize original data files.')
    parser.add_argument('--data_type', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Data type for raw_data mode (train, val, or test)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.mode == 'raw_data':
        output_dir = Path('/workspace/China_Journal/visual')
    else:
        output_dir = Path(args.csv_file).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Heatmap Visualization")
    print("=" * 80)
    
    # Handle raw_data mode separately
    if args.mode == 'raw_data':
        print(f"Mode: raw_data ({args.data_type})")
        print(f"City: {args.city}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        
        output_file = output_dir / f'{args.city}_{args.data_type}_heatmap.png'
        print(f"\nGenerating raw data heatmap from {args.data_type}.csv...")
        visualize_raw_data_heatmap(args.city, args.data_type, str(output_file), args.dpi)
        
        print("\n" + "=" * 80)
        print("Visualization completed!")
        print("=" * 80)
        return
    
    # Prediction result visualization modes
    csv_name = Path(args.csv_file).stem
    print(f"CSV file: {args.csv_file}")
    print(f"City: {args.city}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print("=" * 80)
    
    # Generate visualizations based on mode
    if args.mode in ['ground_truth', 'all']:
        output_file = output_dir / f'{csv_name}_ground_truth_heatmap.png'
        print(f"\n[1] Generating ground truth heatmap...")
        visualize_ground_truth_heatmap(args.csv_file, str(output_file), args.city, args.dpi)
    
    if args.mode in ['comparison', 'all']:
        output_file = output_dir / f'{csv_name}_comparison.png'
        print(f"\n[2] Generating ground truth vs predictions comparison...")
        visualize_prediction_vs_ground_truth(args.csv_file, str(output_file), args.city, args.dpi)
    
    if args.mode in ['top10', 'all']:
        output_file = output_dir / f'{csv_name}_top10_distribution.png'
        print(f"\n[3] Generating top-10 weighted distribution...")
        visualize_top10_distribution(args.csv_file, str(output_file), args.city, args.dpi)
    
    print("\n" + "=" * 80)
    print("All visualizations completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
