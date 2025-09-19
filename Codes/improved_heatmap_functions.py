"""
Revised Heatmap Visualization Functions
This module contains heatmap plotting functions with improved clarity and professional styling.

Key Features:
- Half-diagonal heatmaps for symmetric matrices (co-occurrence vs performance)
- Combined triangular heatmaps showing two metrics in one plot
- Professional styling with larger fonts and clear annotations
- Customizable text annotations via the `show_text` parameter
- Maintains the option to save figures at high quality

Date: 2025-09-07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output (e.g., reindexing alerts)

def _reindex_square(df: pd.DataFrame, labels: list, name: str):
    """Reorder a DataFrame to have rows and columns exactly equal to `labels` (same order)."""
    df2 = df.reindex(index=labels, columns=labels)
    # Check for mismatches and warn if any
    missing_rows = [x for x in labels if x not in df.index]
    missing_cols = [x for x in df.columns if x not in labels]
    extra_rows = [x for x in df.index if x not in labels]
    extra_cols = [x for x in df.columns if x not in labels]
    if missing_rows or missing_cols or extra_rows or extra_cols:
        warnings.warn(
            f"[{name}] Reindexed with differences. "
            f"Missing rows: {missing_rows}; Missing cols: {missing_cols}; "
            f"Extra rows: {extra_rows}; Extra cols: {extra_cols}"
        )
    return df2

def _reindex_columns(df: pd.DataFrame, labels: list, name: str):
    """Reorder a DataFrame's columns to exactly match `labels` (filling missing with NaN)."""
    df2 = df.reindex(columns=labels)
    if any(x not in df.columns for x in labels):
        warnings.warn(f"[{name}] Some requested columns not in DataFrame; filled with NaN.")
    return df2

def plot_half_diagonal_heatmap(cooccur_matrix: pd.DataFrame, performance_matrix: pd.DataFrame,
                               feature_names: list, feature_set_name: str, dataset_name: str,
                               save_path=None, show_text=True, font_size=14):
    """
    Plot a half-diagonal heatmap showing two related matrices side by side:
    - Left: Co-occurrence counts (upper triangle shown)
    - Right: Performance values (upper triangle shown, zeros masked out)

    Args:
        cooccur_matrix (pd.DataFrame): Symmetric co-occurrence matrix.
        performance_matrix (pd.DataFrame): Symmetric performance matrix.
        feature_names (list): List of base feature names (defines matrix order).
        feature_set_name (str): Name of feature set (e.g., 'E23' or 'ALL62').
        dataset_name (str): Name of dataset (e.g., 'GTZAN' or 'MTG-Jamendo').
        save_path (Path or str, optional): If provided, save the figure to this path.
        show_text (bool): Whether to display numerical values in the cells.
        font_size (int): Base font size for plot text.
    """
    # Ensure matrices are reordered to the specified feature order
    cooccur_matrix = _reindex_square(cooccur_matrix, feature_names, "cooccur_matrix")
    performance_matrix = _reindex_square(performance_matrix, feature_names, "performance_matrix")

    n_features = len(feature_names)
    # Figure size: width is double for two subplots, height scales with number of features
    fig_width = max(12, n_features * 0.5) * 2
    fig_height = max(12, n_features * 0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Mask for lower triangle (below diagonal) so only upper triangle data is shown
    mask_lower = np.tril(np.ones_like(cooccur_matrix, dtype=bool), k=-1)

    # Plot 1: Co-occurrence heatmap (upper triangle only)
    sns.heatmap(cooccur_matrix, mask=mask_lower, annot=show_text, fmt='d',
                cmap='YlOrRd', square=True, linewidths=0.5,
                cbar_kws={'label': 'Co-occurrence Count', 'shrink': 0.8}, ax=ax1)
    ax1.set_title(f'{dataset_name} {feature_set_name}: Base Feature Co-occurrence\n(Upper Triangle)',
                  fontsize=font_size+2, fontweight='bold', pad=20)
    ax1.set_xlabel('Base Features', fontsize=14)
    ax1.set_ylabel('Base Features', fontsize=14)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
    ax1.set_yticklabels(feature_names, rotation=0, fontsize=14)
    # Style the colorbar for co-occurrence
    cbar1 = ax1.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=14)
    cbar1.ax.yaxis.label.set_size(14)

    # Plot 2: Performance heatmap (upper triangle only, also mask out zero entries)
    mask_perf = (performance_matrix == 0)  # mask for zero values
    combined_mask = mask_lower | mask_perf  # mask out lower triangle and any zero values
    # Determine color scale range from non-zero performance values for better contrast
    perf_values = performance_matrix.values[performance_matrix.values > 0]
    vmin = perf_values.min() if perf_values.size > 0 else 0
    vmax = perf_values.max() if perf_values.size > 0 else 1
    sns.heatmap(performance_matrix, mask=combined_mask, annot=show_text, fmt='.3f',
                cmap='RdYlBu_r', square=True, linewidths=0.5,
                vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Average val_auc', 'shrink': 0.8}, ax=ax2)
    ax2.set_title(f'{dataset_name} {feature_set_name}: Base Feature Performance\n(Upper Triangle)',
                  fontsize=font_size+2, fontweight='bold', pad=20)
    ax2.set_xlabel('Base Features', fontsize=14)
    ax2.set_ylabel('Base Features', fontsize=14)
    ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
    ax2.set_yticklabels(feature_names, rotation=0, fontsize=14)
    # Style the colorbar for performance
    cbar2 = ax2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=14)
    cbar2.ax.yaxis.label.set_size(14)

    plt.tight_layout()
    # Save and/or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Half-diagonal heatmap saved: {save_path}")
    plt.show()


# def plot_combined_triangular_heatmap(cooccur_matrix: pd.DataFrame, performance_matrix: pd.DataFrame,
#                                      feature_names: list, feature_set_name: str, dataset_name: str,
#                                      save_path=None, show_text=True, font_size=14):
#     """
#     Combined triangular heatmap with dual colorbars.
#     Left colorbar is placed to the far left (outside labels) with automatic spacing.
#     """
#     # Ensure ordering matches feature_names
#     cooccur_matrix = _reindex_square(cooccur_matrix, feature_names, "cooccur_matrix")
#     performance_matrix = _reindex_square(performance_matrix, feature_names, "performance_matrix")

#     n_features = len(feature_names)
#     fig_size = max(12, n_features * 0.4)
#     fig = plt.figure(figsize=(fig_size + 3.5, fig_size))  # a bit wider to afford margins

#     # Provisional main axes (we'll reposition after measuring label width)
#     bottom, height = 0.10, 0.70
#     ax_main = fig.add_axes([0.18, bottom, 0.62, height])

#     # Prepare data & masks
#     cooccur_disp = cooccur_matrix.astype(float).copy()
#     perf_disp = performance_matrix.astype(float).copy()
#     np.fill_diagonal(cooccur_disp.values, np.nan)
#     np.fill_diagonal(perf_disp.values, np.nan)

#     mask_upper = np.triu(np.ones_like(cooccur_disp, dtype=bool), k=0)
#     mask_lower = np.tril(np.ones_like(perf_disp, dtype=bool), k=0)

#     im1 = ax_main.imshow(np.ma.masked_where(mask_upper, cooccur_disp.values),
#                          cmap='YlOrRd', origin='upper')

#     perf_values = performance_matrix.values[performance_matrix.values > 0]
#     vmin_perf = perf_values.min() if perf_values.size > 0 else 0
#     vmax_perf = perf_values.max() if perf_values.size > 0 else 1
#     im2 = ax_main.imshow(np.ma.masked_where(mask_lower, perf_disp.values),
#                          cmap='RdYlBu_r', origin='upper', vmin=vmin_perf, vmax=vmax_perf)

#     # Ticks & labels
#     ax_main.set_xticks(np.arange(n_features))
#     ax_main.set_yticks(np.arange(n_features))
#     ax_main.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=14)
#     ax_main.set_yticklabels(feature_names, rotation=0, fontsize=14)
#     ax_main.set_xticks(np.arange(n_features + 1) - 0.5, minor=True)
#     ax_main.set_yticks(np.arange(n_features + 1) - 0.5, minor=True)
#     ax_main.grid(which='minor', color='white', linewidth=0.5)

#     # --- Measure y-label width and compute positions ---
#     fig.canvas.draw()
#     renderer = fig.canvas.get_renderer()
#     maxw_px = 0
#     for t in ax_main.get_yticklabels():
#         bb = t.get_window_extent(renderer=renderer)
#         maxw_px = max(maxw_px, bb.width)
#     fig_w_px = fig.get_size_inches()[0] * fig.dpi
#     label_frac = maxw_px / fig_w_px  # fraction of figure width used by labels

#     # Layout constants (tweak to taste)
#     left_margin = 0.010          # extra breathing room at extreme left
#     left_cbar_w = 0.032          # width of left colorbar
#     right_cbar_w = 0.030         # width of right colorbar
#     gap_cb_labels = 0.014        # gap between left colorbar and labels
#     label_pad_main = 0.010       # gap between labels and heatmap
#     right_margin = 0.020         # rightmost margin
#     gap_main_rightcb = 0.014     # gap between heatmap and right colorbar

#     # Order from left → right: [left colorbar] gap [labels] pad [main] gap [right colorbar]
#     left_cbar_left = left_margin
#     labels_block_left = left_cbar_left + left_cbar_w + gap_cb_labels
#     main_left = labels_block_left + label_frac + label_pad_main

#     right_cbar_left = 1.0 - right_margin - right_cbar_w
#     main_right = right_cbar_left - gap_main_rightcb
#     main_width = max(0.30, main_right - main_left)

#     # Apply positions
#     ax_main.set_position([main_left, bottom, main_width, height])
#     ax_cbar_left = fig.add_axes([left_cbar_left, bottom, left_cbar_w, height])
#     ax_cbar_right = fig.add_axes([right_cbar_left, bottom, right_cbar_w, height])

#     # Colorbars (using cax -> pad is ignored; positions are explicit)
#     cbar1 = plt.colorbar(im1, cax=ax_cbar_left)
#     cbar1.set_label('Co-occurrence Count', fontsize=14)
#     cbar1.ax.tick_params(labelsize=14)

#     cbar2 = plt.colorbar(im2, cax=ax_cbar_right)
#     cbar2.set_label('Average val_auc', fontsize=14)
#     cbar2.ax.tick_params(labelsize=14)

#     # Titles and labels
#     ax_main.set_title(
#         f'{dataset_name} {feature_set_name}: Combined Base Feature Analysis\n'
#         f'Co-occurrence (Lower) + Performance (Upper)',
#         fontsize=font_size + 2, fontweight='bold', pad=20
#     )
#     ax_main.set_xlabel('Base Features', fontsize=14)
#     ax_main.set_ylabel('Base Features', fontsize=14)

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f"✓ Combined triangular heatmap saved: {save_path}")
#     plt.show()



def plot_combined_triangular_heatmap(cooccur_matrix: pd.DataFrame, performance_matrix: pd.DataFrame,
                                     feature_names: list, feature_set_name: str, dataset_name: str,
                                     save_path=None, show_text=True, font_size=14):
    """
    Plot a combined triangular heatmap on a single axis, where:
    - Lower triangle shows co-occurrence counts
    - Upper triangle shows performance values
    Two separate colorbars (left and right) are used for the different scales.

    Args:
        cooccur_matrix (pd.DataFrame): Symmetric co-occurrence matrix.
        performance_matrix (pd.DataFrame): Symmetric performance matrix.
        feature_names (list): List of base feature names (defines matrix order).
        feature_set_name (str): Name of feature set (e.g., 'E23' or 'ALL62').
        dataset_name (str): Name of dataset (e.g., 'GTZAN' or 'MTG-Jamendo').
        save_path (Path or str, optional): If provided, save the figure to this path.
        show_text (bool): (Reserved) If True, would display values in cells (not recommended for large matrices).
        font_size (int): Base font size for plot text.
    """
    # Ensure matrices use the specified feature order
    cooccur_matrix = _reindex_square(cooccur_matrix, feature_names, "cooccur_matrix")
    performance_matrix = _reindex_square(performance_matrix, feature_names, "performance_matrix")

    n_features = len(feature_names)
    # Figure with main heatmap and two side colorbars
    fig_size = max(12, n_features * 0.4)
    fig = plt.figure(figsize=(fig_size + 3, fig_size))
    # Define axis for main heatmap and colorbars
    ax_main = fig.add_axes([0.185, 0.1, 0.565, 0.7])        # main heatmap in center
    ax_cbar_left = fig.add_axes([0.02, 0.1, 0.03, 0.7])   # left colorbar
    # ax_main = fig.add_axes([0.3, 0.1, 0.45, 0.7])
    # ax_cbar_left = fig.add_axes([0.00, 0.1, 0.03, 0.7])
    ax_cbar_right = fig.add_axes([0.80, 0.1, 0.03, 0.7])  # right colorbar

    # Prepare data: convert to float and mask out diagonal to avoid overlap
    cooccur_disp = cooccur_matrix.astype(float).copy()
    perf_disp = performance_matrix.astype(float).copy()
    for i in range(n_features):
        cooccur_disp.iloc[i, i] = np.nan
        perf_disp.iloc[i, i] = np.nan

    # Co-occurrence: mask upper triangle (including diagonal) and plot lower triangle data
    mask_upper = np.triu(np.ones_like(cooccur_disp, dtype=bool), k=0)
    im1 = ax_main.imshow(np.ma.masked_where(mask_upper, cooccur_disp.values),
                         cmap='YlOrRd', origin='upper')
    # Performance: mask lower triangle (including diagonal) and plot upper triangle data
    mask_lower = np.tril(np.ones_like(perf_disp, dtype=bool), k=0)
    # Determine performance color scale for non-zero values
    perf_values = performance_matrix.values[performance_matrix.values > 0]
    vmin_perf = perf_values.min() if perf_values.size > 0 else 0
    vmax_perf = perf_values.max() if perf_values.size > 0 else 1
    im2 = ax_main.imshow(np.ma.masked_where(mask_lower, perf_disp.values),
                         cmap='RdYlBu_r', origin='upper', vmin=vmin_perf, vmax=vmax_perf)

    # Set tick positions and labels for all features
    ax_main.set_xticks(np.arange(n_features))
    ax_main.set_yticks(np.arange(n_features))
    ax_main.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=15)
    ax_main.set_yticklabels(feature_names, rotation=0, fontsize=15)
    # Draw white grid lines to separate cells
    ax_main.set_xticks(np.arange(n_features+1) - 0.5, minor=True)
    ax_main.set_yticks(np.arange(n_features+1) - 0.5, minor=True)
    ax_main.grid(which='minor', color='white', linewidth=0.5)

    # Create colorbars for each metric
    cbar1 = plt.colorbar(im1, cax=ax_cbar_left, pad=1.5)
    cbar1.set_label('Co-occurrence Count', fontsize=22)
    cbar1.ax.tick_params(labelsize=20)
    cbar2 = plt.colorbar(im2, cax=ax_cbar_right)
    cbar2.set_label('Average val_auc', fontsize=22)
    cbar2.ax.tick_params(labelsize=20)

    # Title and axis labels
    ax_main.set_title(f'{dataset_name} {feature_set_name}: GP Combination of Base Feature Analysis\nCo-occurrence (Lower) + Performance (Upper)',
                      fontsize=font_size+2, fontweight='bold', pad=20)
    ax_main.set_xlabel('Base Features', fontsize=22)
    # ax_main.set_ylabel('Base Features', fontsize=14)

    # Save and/or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Combined triangular heatmap saved: {save_path}")
    plt.show()

def plot_enhanced_heatmap(matrix_df: pd.DataFrame, matrix_type: str,
                          feature_set_name: str, dataset_name: str,
                          save_path=None, show_text=True, font_size=14, colormap=None):
    """
    Plot a single heatmap for a given matrix (co-occurrence or performance) with enhanced styling.

    Args:
        matrix_df (pd.DataFrame): Square matrix of either co-occurrence counts or performance values.
        matrix_type (str): 'Cooccurrence' or 'Performance' (defines color mapping and formatting).
        feature_set_name (str): Name of feature set (e.g., 'E23' or 'ALL62').
        dataset_name (str): Name of dataset (e.g., 'GTZAN' or 'MTG-Jamendo').
        save_path (Path or str, optional): If provided, save the figure to this path.
        show_text (bool): Whether to display numerical values in each cell.
        font_size (int): Base font size for plot text.
        colormap (str, optional): Custom colormap name (overrides default for the given matrix_type).
    """
    n_features = len(matrix_df)
    fig_size = max(12, n_features * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Mask zero values for clarity (will appear blank on the heatmap)
    mask = (matrix_df == 0)
    # Determine settings based on the matrix type
    if matrix_type.lower().startswith('co'):
        cmap = colormap or 'YlOrRd'
        fmt = 'd'
        cbar_label = 'Co-occurrence Count'
        # Co-occurrence counts typically positive integers, use default vmin/vmax
        vmin = None
        vmax = None
        type_label = 'Co-occurrence'
    else:
        cmap = colormap or 'RdYlBu_r'
        fmt = '.4f'
        cbar_label = 'Average val_auc'
        # Set color scale to span the range of non-zero performance values
        perf_vals = matrix_df.values[matrix_df.values > 0]
        if perf_vals.size > 0:
            vmin = perf_vals.min()
            vmax = perf_vals.max()
        else:
            vmin, vmax = 0, 1
        type_label = 'Performance'

    sns.heatmap(matrix_df, annot=show_text, fmt=fmt, cmap=cmap, square=True,
                linewidths=0.5, mask=mask, vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label, 'shrink': 0.8}, ax=ax)
    ax.set_title(f'{dataset_name} {feature_set_name}: Base Feature {type_label} Matrix',
                 fontsize=font_size+4, fontweight='bold', pad=20)
    ax.set_xlabel('Base Features', fontsize=14)
    ax.set_ylabel('Base Features', fontsize=14)
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(matrix_df.index, rotation=0, fontsize=14)
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.label.set_size(14)

    plt.tight_layout()
    # Save and/or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ {type_label} heatmap saved: {save_path}")
    plt.show()

def plot_enhanced_operation_feature_heatmap(matrix_df: pd.DataFrame, matrix_type: str,
                                            feature_set_name: str, dataset_name: str,
                                            save_path=None, show_text=True, font_size=14, colormap=None):
    """
    Plot a heatmap for an operation-feature matrix with enhanced styling.
    Rows represent operations and columns represent base features.

    Args:
        matrix_df (pd.DataFrame): Matrix of shape (operations x base features).
        matrix_type (str): 'Cooccurrence' or 'Performance' (affects color mapping and formatting).
        feature_set_name (str): Name of feature set (e.g., 'E23' or 'ALL62').
        dataset_name (str): Name of dataset (e.g., 'GTZAN' or 'MTG-Jamendo').
        save_path (Path or str, optional): If provided, save the figure to this path.
        show_text (bool): Whether to display numerical values in each cell.
        font_size (int): Base font size for plot text.
        colormap (str, optional): Custom colormap name (overrides default for the given matrix_type).
    """
    n_operations, n_features = matrix_df.shape
    # Figure size scales with number of features (width) and operations (height)
    fig_width = max(14, n_features * 0.4)
    fig_height = max(10, n_operations * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    mask = (matrix_df == 0)
    # Determine settings based on matrix type
    if matrix_type.lower().startswith('co'):
        cmap = colormap or 'YlOrRd'
        fmt = 'd'
        cbar_label = 'Co-occurrence Count'
        vmin = None
        vmax = None
        type_label = 'Co-occurrence'
    else:
        cmap = colormap or 'RdYlBu_r'
        fmt = '.4f'
        cbar_label = 'Average val_auc'
        # Color scale based on non-zero performance values
        perf_vals = matrix_df.values[matrix_df.values > 0]
        if perf_vals.size > 0:
            vmin = perf_vals.min()
            vmax = perf_vals.max()
        else:
            vmin, vmax = 0, 1
        type_label = 'Performance'

    sns.heatmap(matrix_df, annot=show_text, fmt=fmt, cmap=cmap,
                linewidths=0.5, mask=mask, vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label, 'shrink': 0.7}, ax=ax)
    ax.set_title(f'{dataset_name} {feature_set_name}: GP Operation-Feature {type_label} Matrix',
                 fontsize=font_size+3, fontweight='bold', pad=20)
    ax.set_xlabel('Base Features', fontsize=22)
    ax.set_ylabel('Operations', fontsize=22)
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(matrix_df.index, rotation=0, fontsize=18)
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.label.set_size(22)

    plt.tight_layout()
    # Save and/or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Operation-feature {type_label.lower()} heatmap saved: {save_path}")
    plt.show()

def set_publication_style(font_size=12):
    """
    Set global matplotlib parameters for publication-quality visuals.
    Adjusts font sizes, DPI, and other settings for a clean, professional look.
    
    Args:
        font_size (int): Base font size for plots (in points).
    """
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 4,
        'axes.labelsize': font_size + 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size + 6,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5
    })
    # Use a colorblind-friendly palette for consistency
    sns.set_palette("husl")
    print(f"Publication style set with base font size: {font_size}pt")

# If this module is run as a script, print a summary of available functions
if __name__ == "__main__":
    print("Revised Heatmap Visualization Functions")
    print("=" * 50)
    print("Available functions:")
    print("- plot_half_diagonal_heatmap: Split co-occurrence and performance side by side (upper triangles only).")
    print("- plot_combined_triangular_heatmap: Combined co-occurrence (lower) + performance (upper) in one plot.")
    print("- plot_enhanced_heatmap: Single heatmap with enhanced styling for co-occurrence or performance.")
    print("- plot_enhanced_operation_feature_heatmap: Heatmap for operation-feature matrix with enhanced styling.")
    print("- set_publication_style: Configure global styles for publication-quality plots.")
