#!/usr/bin/env python3
"""
Plot results from FFT benchmarks.

Supports both comprehensive_bench.cpp output and bench_mixed_sizes.c output.

Usage:
    ./comprehensive_bench > results.json
    python3 plot_results.py results.json
    
    ./bench_mixed_sizes --json > results.json
    python3 plot_results.py results.json
    
Or run directly:
    ./bench_mixed_sizes --json | python3 plot_results.py
"""

import json
import sys
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)

# Set up matplotlib style
matplotlib.rcParams['figure.figsize'] = (14, 10)
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3

def load_results(source):
    """Load results from file or stdin."""
    if source == '-':
        return json.load(sys.stdin)
    else:
        with open(source, 'r') as f:
            return json.load(f)

def get_libraries(results):
    """Get sorted list of libraries."""
    libs = sorted(set(r['library'] for r in results))
    # Put notoriousfft first for highlighting
    if 'notoriousfft' in libs:
        libs.remove('notoriousfft')
        libs.insert(0, 'notoriousfft')
    return libs

def get_transforms(results):
    """Get list of transforms tested."""
    return sorted(set(r['transform'] for r in results))

def get_size_types(results):
    """Get list of size types (for mixed size benchmarks)."""
    if 'size_type' in results[0]:
        return sorted(set(r['size_type'] for r in results))
    return []

def is_multidimensional_format(results):
    """Check if results are from bench_mixed_sizes (has dim field)."""
    return len(results) > 0 and 'dim' in results[0]

def get_dimension_label(result):
    """Get dimension label for a result."""
    dim = result.get('dim', 1)
    if dim == 1:
        return f"{result['n1']}"
    elif dim == 2:
        return f"{result['n1']}x{result['n2']}"
    elif dim == 3:
        return f"{result['n1']}x{result['n2']}x{result['n3']}"
    elif dim == 4:
        return f"{result['n1']}x{result['n2']}x{result['n3']}x{result['n4']}"
    return str(result.get('N', result['n1']))

def get_total_n(result):
    """Get total number of elements."""
    if 'total_n' in result:
        return result['total_n']
    elif 'N' in result:
        return result['N']
    else:
        dim = result.get('dim', 1)
        n = result['n1']
        if dim >= 2:
            n *= result['n2']
        if dim >= 3:
            n *= result['n3']
        if dim >= 4:
            n *= result['n4']
        return n

def get_performance_metric(result):
    """Get performance metric (GFLOP/s or Mflop/s)."""
    if 'gflops' in result:
        return result['gflops']
    elif 'mflops' in result:
        return result['mflops'] / 1000.0  # Convert to GFLOP/s
    return 0.0

def plot_transform(results, transform, ax_time, ax_perf, colors, markers):
    """Plot results for a single transform type."""
    transform_results = [r for r in results if r['transform'] == transform]
    libraries = get_libraries(transform_results)
    
    for lib in libraries:
        lib_results = [r for r in transform_results if r['library'] == lib]
        lib_results.sort(key=lambda r: get_total_n(r))
        
        Ns = [get_total_n(r) for r in lib_results]
        times = [r['time_us'] for r in lib_results]
        perfs = [get_performance_metric(r) for r in lib_results]
        
        color = colors.get(lib, 'gray')
        marker = markers.get(lib, 'o')
        linewidth = 2.5 if lib == 'notoriousfft' else 1.5
        alpha = 1.0 if lib == 'notoriousfft' else 0.8
        zorder = 10 if lib == 'notoriousfft' else 5
        
        # Time plot
        ax_time.plot(Ns, times, label=lib, color=color, marker=marker, 
                    linewidth=linewidth, markersize=6, alpha=alpha, zorder=zorder)
        
        # Performance plot
        ax_perf.plot(Ns, perfs, label=lib, color=color, marker=marker,
                    linewidth=linewidth, markersize=6, alpha=alpha, zorder=zorder)
    
    ax_time.set_xlabel('Transform Size (N)')
    ax_time.set_ylabel('Time per Transform (μs)')
    ax_time.set_xscale('log', base=2)
    ax_time.set_yscale('log')
    ax_time.set_title(f'{transform.upper()} - Execution Time')
    ax_time.legend(loc='upper left')
    
    ax_perf.set_xlabel('Transform Size (N)')
    ax_perf.set_ylabel('Performance (GFLOP/s)')
    ax_perf.set_xscale('log', base=2)
    ax_perf.set_title(f'{transform.upper()} - Performance')
    ax_perf.legend(loc='upper left')

def create_summary_plot(results, output_path):
    """Create a summary plot with all transforms."""
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    # Define colors and markers
    colors = {
        'notoriousfft': '#e41a1c',    # Red - highlight this one
        'minfft': '#377eb8',    # Blue
        'FFTW3': '#4daf4a',     # Green
        'KissFFT': '#984ea3',   # Purple
        'PocketFFT': '#ff7f00', # Orange
        'muFFT': '#f781bf',     # Pink
    }
    
    markers = {
        'notoriousfft': 'o',
        'minfft': 's',
        'FFTW3': '^',
        'KissFFT': 'D',
        'PocketFFT': 'v',
        'muFFT': 'p',
    }
    
    n_transforms = len(transforms)
    fig, axes = plt.subplots(n_transforms, 2, figsize=(14, 5 * n_transforms))
    
    if n_transforms == 1:
        axes = axes.reshape(1, -1)
    
    for i, transform in enumerate(transforms):
        plot_transform(results, transform, axes[i, 0], axes[i, 1], colors, markers)
    
    plt.suptitle('FFT Benchmark Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    return fig

def create_size_type_plot(results, output_path):
    """Create plot comparing different size types (power2, npo2, prime, mixed)."""
    if not is_multidimensional_format(results):
        return None
    
    size_types = get_size_types(results)
    if not size_types or len(size_types) <= 1:
        return None
    
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    fig, axes = plt.subplots(len(transforms), len(size_types), 
                             figsize=(5 * len(size_types), 4 * len(transforms)))
    
    if len(transforms) == 1 and len(size_types) == 1:
        axes = [[axes]]
    elif len(transforms) == 1:
        axes = [axes]
    elif len(size_types) == 1:
        axes = [[ax] for ax in axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(libraries)))
    color_map = {lib: colors[i] for i, lib in enumerate(libraries)}
    
    for i, transform in enumerate(transforms):
        for j, size_type in enumerate(size_types):
            ax = axes[i][j]
            
            filtered = [r for r in results 
                       if r['transform'] == transform and r.get('size_type') == size_type]
            
            for lib in libraries:
                lib_results = [r for r in filtered if r['library'] == lib]
                # Only 1D results for size type comparison
                lib_results = [r for r in lib_results if r.get('dim', 1) == 1]
                lib_results.sort(key=lambda r: get_total_n(r))
                
                if lib_results:
                    Ns = [get_total_n(r) for r in lib_results]
                    times = [r['time_us'] for r in lib_results]
                    ax.plot(Ns, times, marker='o', label=lib, color=color_map[lib])
            
            ax.set_xlabel('Size (N)')
            ax.set_ylabel('Time (μs)')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_title(f'{transform.upper()} - {size_type}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Performance by Size Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved size type plot to {output_path}")
    
    return fig

def create_dimension_plot(results, output_path):
    """Create plot comparing different dimensionalities."""
    if not is_multidimensional_format(results):
        return None
    
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    # Filter for multi-dimensional results
    md_results = [r for r in results if r.get('dim', 1) > 1]
    if not md_results:
        return None
    
    fig, axes = plt.subplots(1, len(transforms), figsize=(7 * len(transforms), 5))
    if len(transforms) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(libraries)))
    color_map = {lib: colors[i] for i, lib in enumerate(libraries)}
    
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D'}
    
    for idx, transform in enumerate(transforms):
        ax = axes[idx]
        transform_results = [r for r in md_results if r['transform'] == transform]
        
        for lib in libraries:
            lib_results = [r for r in transform_results if r['library'] == lib]
            lib_results.sort(key=lambda r: get_total_n(r))
            
            if lib_results:
                # Group by dimension
                for dim in sorted(set(r.get('dim', 1) for r in lib_results)):
                    dim_results = [r for r in lib_results if r.get('dim', 1) == dim]
                    Ns = [get_total_n(r) for r in dim_results]
                    times = [r['time_us'] for r in dim_results]
                    label = f"{lib} {dim}D" if len(set(r.get('dim', 1) for r in lib_results)) > 1 else lib
                    ax.plot(Ns, times, marker=markers.get(dim, 'o'), 
                           label=label, color=color_map[lib], alpha=0.8)
        
        ax.set_xlabel('Total Elements (N)')
        ax.set_ylabel('Time (μs)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'{transform.upper()} - Multi-Dimensional')
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Dimensional Transform Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved dimension plot to {output_path}")
    
    return fig

def create_speedup_plot(results, output_path):
    """Create speedup plot relative to notoriousfft."""
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    fig, axes = plt.subplots(1, len(transforms), figsize=(6 * len(transforms), 5))
    if len(transforms) == 1:
        axes = [axes]
    
    for idx, transform in enumerate(transforms):
        ax = axes[idx]
        transform_results = [r for r in results if r['transform'] == transform]
        
        # Get notoriousfft baseline
        notorious_fft_results = {get_total_n(r): r['time_us'] 
                         for r in transform_results if r['library'] == 'notoriousfft'}
        
        if not notorious_fft_results:
            ax.text(0.5, 0.5, 'No notoriousfft baseline\nfor comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{transform.upper()} - Speedup vs notoriousfft')
            continue
        
        for lib in libraries:
            if lib == 'notoriousfft':
                continue
            
            lib_results = [r for r in transform_results if r['library'] == lib]
            lib_results.sort(key=lambda r: get_total_n(r))
            
            Ns = []
            speedups = []
            for r in lib_results:
                N = get_total_n(r)
                if N in notorious_fft_results:
                    Ns.append(N)
                    speedups.append(notorious_fft_results[N] / r['time_us'])
            
            if Ns:
                ax.plot(Ns, speedups, marker='o', label=lib, linewidth=1.5)
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='notoriousfft baseline')
        ax.set_xlabel('Transform Size (N)')
        ax.set_ylabel('Speedup vs notoriousfft')
        ax.set_xscale('log', base=2)
        ax.set_title(f'{transform.upper()} - Speedup Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Relative to notoriousfft (>1 means faster than notoriousfft)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved speedup plot to {output_path}")
    
    return fig

def create_normalized_plot(results, output_path):
    """Create normalized performance plot (time divided by N log N)."""
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    colors = {
        'notoriousfft': '#e41a1c',
        'minfft': '#377eb8',
        'FFTW3': '#4daf4a',
        'KissFFT': '#984ea3',
        'PocketFFT': '#ff7f00',
        'muFFT': '#f781bf',
    }
    
    markers = {
        'notoriousfft': 'o',
        'minfft': 's',
        'FFTW3': '^',
        'KissFFT': 'D',
        'PocketFFT': 'v',
        'muFFT': 'p',
    }
    
    fig, axes = plt.subplots(1, len(transforms), figsize=(6 * len(transforms), 5))
    if len(transforms) == 1:
        axes = [axes]
    
    for idx, transform in enumerate(transforms):
        ax = axes[idx]
        transform_results = [r for r in results if r['transform'] == transform]
        
        for lib in libraries:
            lib_results = [r for r in transform_results if r['library'] == lib]
            lib_results.sort(key=lambda r: get_total_n(r))
            
            Ns = [get_total_n(r) for r in lib_results]
            # Normalized time: time / (N log N)
            normalized = [r['time_us'] / (get_total_n(r) * np.log2(get_total_n(r))) 
                         for r in lib_results]
            
            color = colors.get(lib, 'gray')
            marker = markers.get(lib, 'o')
            linewidth = 2.5 if lib == 'notoriousfft' else 1.5
            
            ax.plot(Ns, normalized, label=lib, color=color, marker=marker,
                   linewidth=linewidth, markersize=6)
        
        ax.set_xlabel('Transform Size (N)')
        ax.set_ylabel('Normalized Time (μs / (N log N))')
        ax.set_xscale('log', base=2)
        ax.set_title(f'{transform.upper()} - Normalized Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Normalized Execution Time (Lower is Better)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved normalized plot to {output_path}")
    
    return fig

def print_summary(results):
    """Print text summary of results."""
    transforms = get_transforms(results)
    libraries = get_libraries(results)
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Check if multidimensional
    md_format = is_multidimensional_format(results)
    
    for transform in transforms:
        print(f"\n{transform.upper()}:")
        print("-" * 80)
        
        if md_format:
            print(f"{'Library':<12} {'SizeType':<10} {'Dim':<4} {'Dimensions':<18} {'Time (μs)':>12} {'Perf':>12}")
        else:
            print(f"{'Library':<15} {'N':>10} {'Time (μs)':>12} {'GFLOP/s':>12}")
        
        print("-" * 80)
        
        transform_results = [r for r in results if r['transform'] == transform]
        for lib in libraries:
            lib_results = [r for r in transform_results if r['library'] == lib]
            lib_results.sort(key=lambda r: (r.get('dim', 1), get_total_n(r)))
            
            for r in lib_results:
                if md_format:
                    dims = get_dimension_label(r)
                    size_type = r.get('size_type', 'unknown')
                    dim = r.get('dim', 1)
                    perf = get_performance_metric(r)
                    perf_str = f"{perf*1000:.1f} Mflop/s" if perf < 1 else f"{perf:.2f} GFLOP/s"
                    print(f"{r['library']:<12} {size_type:<10} {dim:<4} {dims:<18} "
                          f"{r['time_us']:>12.2f} {perf_str:>12}")
                else:
                    print(f"{r['library']:<15} {r['N']:>10} {r['time_us']:>12.2f} {r['gflops']:>12.2f}")
    
    print("\n" + "="*80)

def main():
    # Get input source
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Try to read from stdin if piped
        if not sys.stdin.isatty():
            source = '-'
        else:
            print("Usage: python3 plot_results.py <results.json>")
            print("   or: ./bench_mixed_sizes --json | python3 plot_results.py")
            sys.exit(1)
    
    # Load results
    try:
        results = load_results(source)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{source}' not found")
        sys.exit(1)
    
    if not results:
        print("No results to plot")
        sys.exit(1)
    
    # Detect format
    md_format = is_multidimensional_format(results)
    if md_format:
        print(f"Detected multi-dimensional format ({len(results)} results)")
    else:
        print(f"Detected 1D format ({len(results)} results)")
    
    # Print summary
    print_summary(results)
    
    # Determine output directory
    if source == '-':
        output_dir = Path('.')
    else:
        output_dir = Path(source).parent or Path('.')
    
    # Create plots
    try:
        print("\nGenerating plots...")
        
        # Main summary plot
        create_summary_plot(results, output_dir / 'fft_benchmark_summary.png')
        
        # Speedup plot
        create_speedup_plot(results, output_dir / 'fft_benchmark_speedup.png')
        
        # Normalized plot
        create_normalized_plot(results, output_dir / 'fft_benchmark_normalized.png')
        
        # Multi-dimensional specific plots
        if md_format:
            create_size_type_plot(results, output_dir / 'fft_benchmark_size_types.png')
            create_dimension_plot(results, output_dir / 'fft_benchmark_dimensions.png')
        
        print("\nDone! Generated plots:")
        print(f"  - {output_dir / 'fft_benchmark_summary.png'}")
        print(f"  - {output_dir / 'fft_benchmark_speedup.png'}")
        print(f"  - {output_dir / 'fft_benchmark_normalized.png'}")
        if md_format:
            print(f"  - {output_dir / 'fft_benchmark_size_types.png'}")
            print(f"  - {output_dir / 'fft_benchmark_dimensions.png'}")
        
        # Show plots if running interactively
        if sys.stdin.isatty():
            plt.show()
            
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
