"""
Visualize Origin-Destination (OD) flow lines from trajectory data.
Shows movement patterns between locations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import argparse
from pathlib import Path
from collections import defaultdict


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


def grid_id_to_coords(grid_id: int, grid_cols: int) -> tuple:
    """Convert grid ID to (x, y) coordinates (center of grid cell)."""
    row = grid_id // grid_cols
    col = grid_id % grid_cols
    # Return center of grid cell
    x = col + 0.5
    y = row + 0.5
    return x, y


def create_curved_line(start, end, num_points=50, curve_height=0.3):
    """
    Create a curved line between two points using a quadratic Bezier curve.
    
    Args:
        start: (x, y) start point
        end: (x, y) end point
        num_points: Number of points in the curve
        curve_height: Height of the curve (relative to distance)
    
    Returns:
        Array of points along the curve
    """
    x0, y0 = start
    x1, y1 = end
    
    # Calculate midpoint
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2
    
    # Calculate perpendicular direction
    dx = x1 - x0
    dy = y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 0.1:  # Avoid division by zero for very close points
        return np.array([[x0, y0], [x1, y1]])
    
    # Perpendicular vector (normalized)
    px = -dy / dist
    py = dx / dist
    
    # Control point (offset from midpoint)
    offset = dist * curve_height
    cx = mx + px * offset
    cy = my + py * offset
    
    # Generate bezier curve points
    t = np.linspace(0, 1, num_points)
    t = t.reshape(-1, 1)
    
    # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    points = (1-t)**2 * np.array([x0, y0]) + \
             2*(1-t)*t * np.array([cx, cy]) + \
             t**2 * np.array([x1, y1])
    
    return points


def extract_od_pairs_from_raw_data(csv_file: str, max_pairs: int = None) -> dict:
    """
    Extract OD pairs from raw trajectory data (train.csv, val.csv, test.csv).
    
    Args:
        csv_file: Path to CSV file
        max_pairs: Maximum number of pairs to extract (for large datasets)
    
    Returns:
        Dictionary with OD pair counts: {(origin, destination): count}
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} records")
    
    od_pairs = defaultdict(int)
    total_pairs = 0
    
    print("Extracting OD pairs from trajectories...")
    for user_id, group in df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # Extract consecutive location pairs
        for i in range(len(group) - 1):
            origin = int(group.iloc[i]['location_id'])
            destination = int(group.iloc[i + 1]['location_id'])
            
            # Skip self-loops if desired (staying at same location)
            # if origin == destination:
            #     continue
            
            od_pairs[(origin, destination)] += 1
            total_pairs += 1
            
            if max_pairs and total_pairs >= max_pairs:
                break
        
        if max_pairs and total_pairs >= max_pairs:
            break
    
    print(f"Extracted {len(od_pairs)} unique OD pairs ({total_pairs} total movements)")
    
    return dict(od_pairs)


def extract_od_pairs_from_predictions(csv_file: str) -> dict:
    """
    Extract OD pairs from prediction results CSV.
    Uses observation trajectory (last location) as origin and ground truth as destination.
    
    Args:
        csv_file: Path to predictions CSV file
    
    Returns:
        Dictionary with OD pair counts: {(origin, destination): count}
    """
    print(f"Loading predictions from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} predictions")
    
    od_pairs = defaultdict(int)
    
    print("Extracting OD pairs from predictions...")
    for _, row in df.iterrows():
        origin = int(row['obs_loc_-1'])  # Last observation location
        destination = int(row['ground_truth'])  # Ground truth next location
        
        od_pairs[(origin, destination)] += 1
    
    print(f"Extracted {len(od_pairs)} unique OD pairs")
    
    return dict(od_pairs)


def visualize_od_lines(
    od_pairs: dict,
    city: str,
    output_file: str = None,
    dpi: int = 300,
    min_flow: int = 1,
    max_lines: int = None,
    show_self_loops: bool = False,
    line_width_scale: float = 1.0,
    alpha: float = 0.5,
    use_curved_lines: bool = True,
    show_points: bool = True,
    line_color: str = 'purple'
):
    """
    Visualize OD pairs as lines on a grid (curved style like reference image).
    
    Args:
        od_pairs: Dictionary {(origin, destination): count}
        city: City name
        output_file: Output file path
        dpi: DPI for output image
        min_flow: Minimum flow to display
        max_lines: Maximum number of lines to display (top flows)
        show_self_loops: Whether to show self-loops (origin == destination)
        line_width_scale: Scale factor for line width
        alpha: Transparency of lines
        use_curved_lines: Use curved lines instead of straight
        show_points: Show origin (green) and destination (white) points
        line_color: Color scheme ('purple', 'red', or matplotlib colormap)
    """
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    
    # Filter OD pairs
    filtered_pairs = {}
    for (origin, dest), count in od_pairs.items():
        if count < min_flow:
            continue
        if not show_self_loops and origin == dest:
            continue
        filtered_pairs[(origin, dest)] = count
    
    # Sort by flow and limit
    sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)
    
    if max_lines and len(sorted_pairs) > max_lines:
        print(f"Limiting to top {max_lines} flows out of {len(sorted_pairs)}")
        sorted_pairs = sorted_pairs[:max_lines]
    
    print(f"Visualizing {len(sorted_pairs)} OD pairs")
    
    # Prepare line data and extract origins/destinations
    lines = []
    flows = []
    origins = set()
    destinations = set()
    
    for (origin, dest), count in sorted_pairs:
        ox, oy = grid_id_to_coords(origin, grid_cols)
        dx, dy = grid_id_to_coords(dest, grid_cols)
        
        origins.add((ox, oy))
        destinations.add((dx, dy))
        
        if use_curved_lines:
            # Create curved line
            curve_points = create_curved_line((ox, oy), (dx, dy), num_points=30, curve_height=0.15)
            lines.append(curve_points)
        else:
            lines.append([(ox, oy), (dx, dy)])
        
        flows.append(count)
    
    if not lines:
        print("No lines to display!")
        return
    
    flows = np.array(flows)
    
    # Normalize flows for visualization
    min_flow_val = flows.min()
    max_flow_val = flows.max()
    
    print(f"Flow range: {min_flow_val} - {max_flow_val}")
    
    # Create figure with light background
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#f5f5f5')
    
    # Normalize line widths (0.3 to 2.5 for better visibility)
    if max_flow_val > min_flow_val:
        line_widths = 0.3 + (flows - min_flow_val) / (max_flow_val - min_flow_val) * 2.2
    else:
        line_widths = np.ones_like(flows) * 1.0
    
    line_widths *= line_width_scale
    
    # Determine color scheme
    if line_color == 'purple':
        colors = ['#8B008B', '#9932CC', '#BA55D3']  # Dark purple to medium purple
    elif line_color == 'red':
        colors = ['#DC143C', '#FF1493', '#FF69B4']  # Crimson to hot pink
    else:
        colors = None
    
    # Draw lines
    for i, (line, width) in enumerate(zip(lines, line_widths)):
        if colors:
            # Vary color slightly for depth
            color = colors[i % len(colors)]
        else:
            color = plt.cm.plasma(flows[i] / max_flow_val)
        
        if use_curved_lines:
            ax.plot(line[:, 0], line[:, 1], color=color, alpha=alpha, 
                   linewidth=width, solid_capstyle='round', zorder=1)
        else:
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                   color=color, alpha=alpha, linewidth=width, 
                   solid_capstyle='round', zorder=1)
    
    # Draw origin and destination points
    if show_points:
        # Origins (green)
        origins_array = np.array(list(origins))
        if len(origins_array) > 0:
            ax.scatter(origins_array[:, 0], origins_array[:, 1], 
                      c='#00FF00', s=50, alpha=0.7, edgecolors='darkgreen',
                      linewidths=1, zorder=3, label='Origins')
        
        # Destinations (white with border)
        destinations_array = np.array(list(destinations))
        if len(destinations_array) > 0:
            ax.scatter(destinations_array[:, 0], destinations_array[:, 1],
                      c='white', s=40, alpha=0.9, edgecolors='gray',
                      linewidths=1, zorder=2, label='Destinations')
    
    # Set axis limits and labels
    ax.set_xlim(-0.5, grid_cols + 0.5)
    ax.set_ylim(-0.5, grid_rows + 0.5)
    ax.set_xlabel('Grid Column', fontsize=12, fontweight='bold')
    ax.set_ylabel('Grid Row', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    
    # Add subtle grid
    ax.grid(True, alpha=0.15, linewidth=0.5, color='gray', linestyle='--')
    ax.set_xticks(np.arange(0, grid_cols + 1, 5))
    ax.set_yticks(np.arange(0, grid_rows + 1, 5))
    
    # Title
    total_flow = flows.sum()
    ax.set_title(
        f'Origin-Destination Flow Pattern - {city.title()}\n'
        f'OD Pairs: {len(sorted_pairs):,} | Total Movements: {int(total_flow):,}',
        fontsize=15, fontweight='bold', pad=20
    )
    
    # Add legend if points are shown
    if show_points:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistics text box
    stats_text = (
        f'Flow Statistics:\n'
        f'  Min: {int(min_flow_val)}\n'
        f'  Max: {int(max_flow_val)}\n'
        f'  Avg: {flows.mean():.1f}\n'
        f'  Median: {np.median(flows):.1f}'
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        print(f"\nSaving OD visualization to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def visualize_od_with_arrows(
    od_pairs: dict,
    city: str,
    output_file: str = None,
    dpi: int = 300,
    min_flow: int = 1,
    max_arrows: int = 500,
    show_self_loops: bool = False,
    arrow_width_scale: float = 0.003,
    alpha: float = 0.5
):
    """
    Visualize OD pairs as arrows showing direction.
    
    Args:
        od_pairs: Dictionary {(origin, destination): count}
        city: City name
        output_file: Output file path
        dpi: DPI for output image
        min_flow: Minimum flow to display
        max_arrows: Maximum number of arrows to display
        show_self_loops: Whether to show self-loops
        arrow_width_scale: Scale factor for arrow width
        alpha: Transparency of arrows
    """
    # Get grid dimensions
    grid_rows, grid_cols = load_grid_dimensions(city)
    
    # Filter OD pairs
    filtered_pairs = {}
    for (origin, dest), count in od_pairs.items():
        if count < min_flow:
            continue
        if not show_self_loops and origin == dest:
            continue
        filtered_pairs[(origin, dest)] = count
    
    # Sort by flow and limit
    sorted_pairs = sorted(filtered_pairs.items(), key=lambda x: x[1], reverse=True)
    
    if max_arrows and len(sorted_pairs) > max_arrows:
        print(f"Limiting to top {max_arrows} flows out of {len(sorted_pairs)}")
        sorted_pairs = sorted_pairs[:max_arrows]
    
    print(f"Visualizing {len(sorted_pairs)} OD pairs as arrows")
    
    if not sorted_pairs:
        print("No arrows to display!")
        return
    
    # Get flows for normalization
    flows = np.array([count for _, count in sorted_pairs])
    min_flow_val = flows.min()
    max_flow_val = flows.max()
    
    print(f"Flow range: {min_flow_val} - {max_flow_val}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot arrows
    for (origin, dest), count in sorted_pairs:
        ox, oy = grid_id_to_coords(origin, grid_cols)
        dx, dy = grid_id_to_coords(dest, grid_cols)
        
        # Calculate arrow direction
        arrow_dx = dx - ox
        arrow_dy = dy - oy
        
        # Normalize arrow width based on flow
        if max_flow_val > min_flow_val:
            width = arrow_width_scale * (1 + 4 * (count - min_flow_val) / (max_flow_val - min_flow_val))
        else:
            width = arrow_width_scale * 2
        
        # Normalize color based on flow
        color_intensity = (count - min_flow_val) / (max_flow_val - min_flow_val) if max_flow_val > min_flow_val else 0.5
        color = plt.cm.plasma(color_intensity)
        
        # Draw arrow
        ax.arrow(ox, oy, arrow_dx, arrow_dy,
                head_width=width * 3, head_length=width * 2,
                fc=color, ec=color, alpha=alpha,
                length_includes_head=True, linewidth=width * 200)
    
    # Set axis limits and labels
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_xlabel('Grid Column', fontsize=12)
    ax.set_ylabel('Grid Row', fontsize=12)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xticks(np.arange(0, grid_cols + 1, 5))
    ax.set_yticks(np.arange(0, grid_rows + 1, 5))
    
    # Title
    total_flow = flows.sum()
    ax.set_title(
        f'Origin-Destination Flow Arrows - {city.title()}\n'
        f'OD Pairs: {len(sorted_pairs):,} | Total Flow: {int(total_flow):,}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min_flow_val, vmax=max_flow_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flow Count', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        print(f"\nSaving OD arrow visualization to {output_file}...")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved successfully!")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize OD flow lines from trajectory data')
    parser.add_argument('--csv_file', type=str, default='/workspace/China_Journal/visual/test_for_visual_beijing.csv',
                       help='Path to CSV file (train.csv, predictions.csv, etc.)')
    parser.add_argument('--city', type=str, default='beijing',
                       choices=['beijing', 'shenzhen', 'nanchang'],
                       help='City name for grid dimensions')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for saved images. Default: visual/')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for output images')
    parser.add_argument('--mode', type=str, default='lines',
                       choices=['lines', 'arrows', 'both'],
                       help='Visualization mode: lines or arrows')
    parser.add_argument('--data_type', type=str, default='raw_data',
                       choices=['raw_data', 'predictions'],
                       help='Type of input data: raw (train/val/test.csv) or predictions')
    parser.add_argument('--min_flow', type=int, default=1,
                       help='Minimum flow count to display')
    parser.add_argument('--max_lines', type=int, default=1000,
                       help='Maximum number of lines/arrows to display')
    parser.add_argument('--show_self_loops', action='store_true', default=False,
                       help='Show self-loops (staying at same location)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Line/arrow transparency (0-1)')
    parser.add_argument('--line_width_scale', type=float, default=1.0,
                       help='Scale factor for line width')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum OD pairs to extract from raw data (for large datasets)')
    parser.add_argument('--use_curved_lines', action='store_true', default=True,
                       help='Use curved lines (default: True)')
    parser.add_argument('--straight_lines', action='store_true',
                       help='Use straight lines instead of curved')
    parser.add_argument('--no_points', action='store_true',
                       help='Do not show origin/destination points')
    parser.add_argument('--line_color', type=str, default='purple',
                       choices=['purple', 'red', 'plasma', 'viridis'],
                       help='Line color scheme')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('/workspace/China_Journal/visual')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("OD Flow Visualization")
    print("=" * 80)
    print(f"CSV file: {args.csv_file}")
    print(f"City: {args.city}")
    print(f"Data type: {args.data_type}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Extract OD pairs
    print(f"\n[1/2] Extracting OD pairs...")
    if args.data_type == 'predictions':
        od_pairs = extract_od_pairs_from_predictions(args.csv_file)
    else:
        od_pairs = extract_od_pairs_from_raw_data(args.csv_file, args.max_pairs)
    
    # Generate visualizations
    print(f"\n[2/2] Generating visualizations...")
    
    csv_name = Path(args.csv_file).stem
    
    if args.mode in ['lines', 'both']:
        output_file = output_dir / f'{csv_name}_od_lines.png'
        print(f"\nGenerating line visualization...")
        use_curved = not args.straight_lines
        visualize_od_lines(
            od_pairs=od_pairs,
            city=args.city,
            output_file=str(output_file),
            dpi=args.dpi,
            min_flow=args.min_flow,
            max_lines=args.max_lines,
            show_self_loops=args.show_self_loops,
            line_width_scale=args.line_width_scale,
            alpha=args.alpha,
            use_curved_lines=use_curved,
            show_points=not args.no_points,
            line_color=args.line_color
        )
    
    if args.mode in ['arrows', 'both']:
        output_file = output_dir / f'{csv_name}_od_arrows.png'
        print(f"\nGenerating arrow visualization...")
        visualize_od_with_arrows(
            od_pairs=od_pairs,
            city=args.city,
            output_file=str(output_file),
            dpi=args.dpi,
            min_flow=args.min_flow,
            max_arrows=args.max_lines,
            show_self_loops=args.show_self_loops,
            alpha=args.alpha
        )
    
    print("\n" + "=" * 80)
    print("OD visualization completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
