"""
Paper-Ready Visualization System for NSRPO
Task 14: Build Visualization and Plotting - Paper-ready figures and charts

Publication-quality visualization system for NSRPO evaluation results.
Creates IEEE/ACM conference standard figures with proper formatting,
statistical annotations, and vector graphics output.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Statistical visualization
from scipy import stats
import matplotlib.patches as mpatches

# Configuration for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# IEEE/ACM paper formatting
PAPER_CONFIG = {
    'figure_width': 3.5,  # inches (single column)
    'figure_width_double': 7.16,  # inches (double column) 
    'figure_height': 2.5,  # inches
    'dpi': 300,
    'font_size': 9,
    'label_size': 8,
    'legend_size': 7,
    'line_width': 1.2,
    'marker_size': 4,
    'formats': ['pdf', 'png', 'eps']  # Vector formats for papers
}

# Color schemes for different plot types
COLOR_SCHEMES = {
    'comparison': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
    'ablation': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
    'metrics': ['#355070', '#6D597A', '#B56576', '#E56B6F', '#EAAC8B'],
    'performance': ['#0077B6', '#0096C7', '#00B4D8', '#48CAE4', '#90E0EF']
}


class PaperPlotGenerator:
    """
    Publication-quality plot generator for NSRPO evaluation results.
    
    Creates IEEE/ACM standard figures with proper formatting, statistical
    annotations, and multiple output formats.
    """
    
    def __init__(
        self,
        output_dir: str = "./paper_figures",
        style: str = "ieee",
        color_scheme: str = "comparison"
    ):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory to save figures
            style: Plot style ('ieee', 'acm', 'nature')
            color_scheme: Color scheme to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.style = style
        self.colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['comparison'])
        
        # Configure matplotlib for publication quality
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality plots."""
        plt.rcParams.update({
            'font.size': PAPER_CONFIG['font_size'],
            'axes.labelsize': PAPER_CONFIG['label_size'],
            'axes.titlesize': PAPER_CONFIG['font_size'],
            'xtick.labelsize': PAPER_CONFIG['label_size'],
            'ytick.labelsize': PAPER_CONFIG['label_size'],
            'legend.fontsize': PAPER_CONFIG['legend_size'],
            'lines.linewidth': PAPER_CONFIG['line_width'],
            'lines.markersize': PAPER_CONFIG['marker_size'],
            'figure.dpi': PAPER_CONFIG['dpi'],
            'savefig.dpi': PAPER_CONFIG['dpi'],
            'font.family': 'serif',
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        })
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'perplexity', 'kl_divergence'],
        title: str = "Model Performance Comparison",
        save_name: str = "model_comparison",
        show_confidence_intervals: bool = True,
        statistical_tests: Optional[Dict] = None
    ) -> str:
        """
        Create model comparison plot.
        
        Args:
            results: Dictionary with model_name -> {metric: value} mappings
            metrics: List of metrics to plot
            title: Plot title
            save_name: Filename to save (without extension)
            show_confidence_intervals: Whether to show error bars
            statistical_tests: Statistical test results for annotations
            
        Returns:
            Path to saved figure
        """
        n_metrics = len(metrics)
        
        # Create subplots
        fig_width = PAPER_CONFIG['figure_width_double'] if n_metrics > 2 else PAPER_CONFIG['figure_width']
        fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, PAPER_CONFIG['figure_height']))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values for this metric
            values = []
            errors = []
            
            for model_name in model_names:
                if metric in results[model_name]:
                    value = results[model_name][metric]
                    if isinstance(value, dict) and 'mean' in value:
                        values.append(value['mean'])
                        errors.append(value.get('std', 0))
                    else:
                        values.append(float(value))
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Create bar plot
            x_pos = np.arange(len(model_names))
            bars = ax.bar(x_pos, values, yerr=errors if show_confidence_intervals else None,
                         capsize=3, color=self.colors[:len(model_names)],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Model')
            ax.set_ylabel(self._format_metric_name(metric))
            ax.set_title(f'{self._format_metric_name(metric)}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([self._format_model_name(name) for name in model_names], 
                              rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=7)
            
            # Add statistical significance annotations
            if statistical_tests and metric in statistical_tests:
                self._add_significance_annotations(ax, statistical_tests[metric], x_pos, values, errors)
            
            ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.suptitle(title, fontsize=PAPER_CONFIG['font_size'] + 1, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save figure
        return self._save_figure(fig, save_name)
    
    def plot_training_curves(
        self,
        training_logs: Dict[str, List[Dict]],
        metrics: List[str] = ['loss', 'accuracy'],
        title: str = "Training Curves",
        save_name: str = "training_curves",
        smoothing: bool = True
    ) -> str:
        """
        Create training curves plot.
        
        Args:
            training_logs: Dictionary with model_name -> list of log entries
            metrics: Metrics to plot
            title: Plot title
            save_name: Filename to save
            smoothing: Whether to apply smoothing to curves
            
        Returns:
            Path to saved figure
        """
        n_metrics = len(metrics)
        
        fig_width = PAPER_CONFIG['figure_width_double']
        fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, PAPER_CONFIG['figure_height']))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, (model_name, logs) in enumerate(training_logs.items()):
                # Extract metric values and steps
                steps = []
                values = []
                
                for log_entry in logs:
                    if 'step' in log_entry and metric in log_entry:
                        steps.append(log_entry['step'])
                        values.append(log_entry[metric])
                
                if not steps:
                    continue
                
                steps = np.array(steps)
                values = np.array(values)
                
                # Apply smoothing if requested
                if smoothing and len(values) > 10:
                    values = self._smooth_curve(values, window_size=min(len(values)//10, 10))
                
                # Plot curve
                ax.plot(steps, values, label=self._format_model_name(model_name),
                       color=self.colors[j % len(self.colors)], linewidth=1.5,
                       marker='o', markersize=2, markevery=max(len(steps)//20, 1))
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel(self._format_metric_name(metric))
            ax.set_title(f'{self._format_metric_name(metric)}')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=PAPER_CONFIG['font_size'] + 1, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return self._save_figure(fig, save_name)
    
    def plot_ablation_study(
        self,
        ablation_results: List[Dict],
        component_name: str,
        metric: str = 'accuracy',
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        show_baseline: bool = True
    ) -> str:
        """
        Create ablation study plot.
        
        Args:
            ablation_results: List of ablation results
            component_name: Name of component being ablated
            metric: Metric to plot
            title: Plot title
            save_name: Filename to save
            show_baseline: Whether to highlight baseline value
            
        Returns:
            Path to saved figure
        """
        if title is None:
            title = f"Ablation Study: {self._format_component_name(component_name)}"
        
        if save_name is None:
            save_name = f"ablation_{component_name}_{metric}"
        
        fig, ax = plt.subplots(figsize=(PAPER_CONFIG['figure_width'], PAPER_CONFIG['figure_height']))
        
        # Extract data
        component_values = []
        metric_values = []
        baseline_idx = None
        
        for i, result in enumerate(ablation_results):
            if result['component_name'] == component_name:
                component_values.append(result['component_value'])
                metric_values.append(result['metrics'][metric])
                
                # Check if this is baseline
                if show_baseline and result.get('is_baseline', False):
                    baseline_idx = len(component_values) - 1
        
        if not component_values:
            raise ValueError(f"No results found for component: {component_name}")
        
        # Sort by component value
        sorted_indices = np.argsort(component_values)
        component_values = [component_values[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
        
        if baseline_idx is not None:
            baseline_idx = sorted_indices.tolist().index(baseline_idx)
        
        # Create plot
        x_pos = np.arange(len(component_values))
        bars = ax.bar(x_pos, metric_values, color=self.colors[1], alpha=0.7,
                     edgecolor='black', linewidth=0.5)
        
        # Highlight baseline if present
        if baseline_idx is not None:
            bars[baseline_idx].set_color(self.colors[0])
            bars[baseline_idx].set_alpha(1.0)
            bars[baseline_idx].set_linewidth(2)
        
        # Customize plot
        ax.set_xlabel(self._format_component_name(component_name))
        ax.set_ylabel(self._format_metric_name(metric))
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in component_values], rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ax.get_ylim()[1]*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        # Add baseline indicator
        if baseline_idx is not None:
            ax.text(baseline_idx, metric_values[baseline_idx] + ax.get_ylim()[1]*0.05,
                   'Baseline', ha='center', va='bottom', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_performance_heatmap(
        self,
        results_matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Performance Heatmap",
        save_name: str = "performance_heatmap",
        metric_name: str = "Score",
        cmap: str = 'RdYlBu_r'
    ) -> str:
        """
        Create performance heatmap.
        
        Args:
            results_matrix: 2D array of results
            row_labels: Labels for rows (e.g., models)
            col_labels: Labels for columns (e.g., metrics)
            title: Plot title
            save_name: Filename to save
            metric_name: Name of metric for colorbar
            cmap: Colormap to use
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(PAPER_CONFIG['figure_width'], PAPER_CONFIG['figure_height']))
        
        # Create heatmap
        im = ax.imshow(results_matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
        ax.set_yticklabels(row_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name, rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{results_matrix[i, j]:.3f}',
                              ha='center', va='center', color='black',
                              fontsize=7)
        
        ax.set_title(title)
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_statistical_comparison(
        self,
        group_data: Dict[str, List[float]],
        title: str = "Statistical Comparison",
        save_name: str = "statistical_comparison",
        test_results: Optional[Dict] = None,
        plot_type: str = 'boxplot'
    ) -> str:
        """
        Create statistical comparison plot with significance annotations.
        
        Args:
            group_data: Dictionary with group_name -> list of values
            title: Plot title
            save_name: Filename to save
            test_results: Statistical test results
            plot_type: Type of plot ('boxplot', 'violinplot', 'barplot')
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(PAPER_CONFIG['figure_width'], PAPER_CONFIG['figure_height']))
        
        groups = list(group_data.keys())
        data = list(group_data.values())
        
        if plot_type == 'boxplot':
            bp = ax.boxplot(data, labels=groups, patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        elif plot_type == 'violinplot':
            vp = ax.violinplot(data, positions=range(1, len(groups)+1))
            for pc, color in zip(vp['bodies'], self.colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax.set_xticks(range(1, len(groups)+1))
            ax.set_xticklabels(groups)
        
        elif plot_type == 'barplot':
            means = [np.mean(d) for d in data]
            stds = [np.std(d) for d in data]
            x_pos = np.arange(len(groups))
            
            bars = ax.bar(x_pos, means, yerr=stds, capsize=3,
                         color=self.colors[:len(groups)], alpha=0.7,
                         edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups)
        
        # Add statistical significance annotations
        if test_results:
            self._add_pairwise_significance(ax, test_results, groups, data)
        
        ax.set_xlabel('Groups')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        labels: List[str],
        title: str = "Metric Correlations",
        save_name: str = "correlation_matrix",
        annot: bool = True
    ) -> str:
        """
        Create correlation matrix heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            labels: Labels for metrics
            title: Plot title
            save_name: Filename to save
            annot: Whether to annotate cells with values
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(PAPER_CONFIG['figure_width'], PAPER_CONFIG['figure_width']))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show only lower triangle
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                      mask=mask, aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # Add annotations
        if annot:
            for i in range(len(labels)):
                for j in range(i):
                    text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                  ha='center', va='center',
                                  color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                                  fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        
        return self._save_figure(fig, save_name)
    
    def create_figure_summary(
        self,
        results: Dict[str, Any],
        save_name: str = "figure_summary"
    ) -> str:
        """
        Create a multi-panel summary figure.
        
        Args:
            results: Dictionary containing all results
            save_name: Filename to save
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(PAPER_CONFIG['figure_width_double'], 
                                PAPER_CONFIG['figure_height'] * 2))
        
        # Create grid layout
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Model comparison
        if 'model_comparison' in results:
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_mini_comparison_plot(ax1, results['model_comparison'])
            ax1.set_title('(a) Model Comparison', fontsize=9, weight='bold')
        
        # Panel B: Training curves
        if 'training_curves' in results:
            ax2 = fig.add_subplot(gs[0, 1])
            self._create_mini_training_plot(ax2, results['training_curves'])
            ax2.set_title('(b) Training Progress', fontsize=9, weight='bold')
        
        # Panel C: Ablation study
        if 'ablation_results' in results:
            ax3 = fig.add_subplot(gs[0, 2])
            self._create_mini_ablation_plot(ax3, results['ablation_results'])
            ax3.set_title('(c) Ablation Study', fontsize=9, weight='bold')
        
        # Panel D: Performance heatmap
        if 'performance_matrix' in results:
            ax4 = fig.add_subplot(gs[1, :2])
            self._create_mini_heatmap(ax4, results['performance_matrix'])
            ax4.set_title('(d) Performance Matrix', fontsize=9, weight='bold')
        
        # Panel E: Statistical analysis
        if 'statistical_tests' in results:
            ax5 = fig.add_subplot(gs[1, 2])
            self._create_mini_stats_plot(ax5, results['statistical_tests'])
            ax5.set_title('(e) Statistical Tests', fontsize=9, weight='bold')
        
        plt.suptitle('NSRPO Evaluation Results', fontsize=12, weight='bold', y=0.98)
        
        return self._save_figure(fig, save_name)
    
    def _create_mini_comparison_plot(self, ax, data):
        """Create mini comparison plot for summary figure."""
        # Simplified version of model comparison
        models = list(data.keys())
        values = [data[m]['accuracy'] for m in models]
        
        bars = ax.bar(range(len(models)), values, color=self.colors[:len(models)], alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:8] for m in models], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_mini_training_plot(self, ax, data):
        """Create mini training plot for summary figure."""
        for i, (model_name, logs) in enumerate(data.items()):
            steps = [log['step'] for log in logs if 'step' in log and 'loss' in log]
            losses = [log['loss'] for log in logs if 'step' in log and 'loss' in log]
            
            if steps:
                ax.plot(steps, losses, label=model_name[:8], color=self.colors[i], linewidth=1)
        
        ax.set_xlabel('Steps', fontsize=8)
        ax.set_ylabel('Loss', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    def _create_mini_ablation_plot(self, ax, data):
        """Create mini ablation plot for summary figure."""
        # Show importance scores
        components = [d['component_name'] for d in data if 'importance_score' in d][:5]
        scores = [d['importance_score'] for d in data if 'importance_score' in d][:5]
        
        bars = ax.barh(range(len(components)), scores, color=self.colors[2], alpha=0.7)
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels([c[:10] for c in components], fontsize=7)
        ax.set_xlabel('Importance', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_mini_heatmap(self, ax, data):
        """Create mini heatmap for summary figure."""
        matrix = data['matrix']
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_xticks(range(len(data['col_labels'])))
        ax.set_yticks(range(len(data['row_labels'])))
        ax.set_xticklabels(data['col_labels'], rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(data['row_labels'], fontsize=7)
        
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    def _create_mini_stats_plot(self, ax, data):
        """Create mini statistics plot for summary figure."""
        # Show p-values
        tests = list(data.keys())[:5]
        p_values = [data[t]['p_value'] for t in tests]
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars = ax.barh(range(len(tests)), [-np.log10(p) for p in p_values], 
                      color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(tests)))
        ax.set_yticklabels([t[:10] for t in tests], fontsize=7)
        ax.set_xlabel('-log10(p)', fontsize=8)
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
    
    def _smooth_curve(self, values: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing to curve."""
        return np.convolve(values, np.ones(window_size)/window_size, mode='same')
    
    def _add_significance_annotations(self, ax, test_results, x_pos, values, errors):
        """Add statistical significance annotations to plot."""
        if 'comparisons' not in test_results:
            return
        
        y_max = max(values) + max(errors) if errors else max(values)
        y_offset = y_max * 0.1
        
        for i, comparison in enumerate(test_results['comparisons']):
            if comparison['p_value'] < 0.05:
                x1, x2 = comparison['indices']
                y = y_max + y_offset * (i + 1)
                
                # Draw significance line
                ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
                ax.plot([x1, x1], [y - y_offset*0.1, y], 'k-', linewidth=1)
                ax.plot([x2, x2], [y - y_offset*0.1, y], 'k-', linewidth=1)
                
                # Add significance stars
                if comparison['p_value'] < 0.001:
                    sig_text = '***'
                elif comparison['p_value'] < 0.01:
                    sig_text = '**'
                else:
                    sig_text = '*'
                
                ax.text((x1 + x2) / 2, y + y_offset*0.1, sig_text,
                       ha='center', va='bottom', fontsize=8)
    
    def _add_pairwise_significance(self, ax, test_results, groups, data):
        """Add pairwise significance annotations."""
        y_max = max(max(d) for d in data)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.05
        
        for i, (comparison, result) in enumerate(test_results.items()):
            if result['p_value'] < 0.05:
                # Parse comparison (e.g., "Group1_vs_Group2")
                group1, group2 = comparison.split('_vs_')
                x1 = groups.index(group1)
                x2 = groups.index(group2)
                
                y = y_max + y_offset * (i + 1)
                
                # Draw significance line
                ax.plot([x1+1, x2+1], [y, y], 'k-', linewidth=1)
                
                # Add significance indicator
                if result['p_value'] < 0.001:
                    sig_text = '***'
                elif result['p_value'] < 0.01:
                    sig_text = '**'
                else:
                    sig_text = '*'
                
                ax.text((x1 + x2 + 2) / 2, y + y_offset*0.2, sig_text,
                       ha='center', va='bottom', fontsize=8)
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        format_map = {
            'accuracy': 'Accuracy',
            'perplexity': 'Perplexity',
            'kl_divergence': 'KL Divergence',
            'loss': 'Loss',
            'f1_score': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        return format_map.get(metric, metric.replace('_', ' ').title())
    
    def _format_model_name(self, model: str) -> str:
        """Format model name for display."""
        return model.replace('_', ' ').replace('nsrpo', 'NSRPO').replace('grpo', 'GRPO')
    
    def _format_component_name(self, component: str) -> str:
        """Format component name for display."""
        format_map = {
            'alpha_1_ce_loss': 'α₁ (CE Loss)',
            'alpha_2_cosine_loss': 'α₂ (Cosine)',
            'alpha_3_norm_preservation': 'α₃ (Norm)',
            'decoder_layers': 'Decoder Layers',
            'decoder_heads': 'Attention Heads',
            'learning_rate': 'Learning Rate'
        }
        return format_map.get(component, component.replace('_', ' ').title())
    
    def _save_figure(self, fig, save_name: str) -> str:
        """Save figure in multiple formats."""
        saved_paths = []
        
        for fmt in PAPER_CONFIG['formats']:
            file_path = self.output_dir / f"{save_name}.{fmt}"
            fig.savefig(file_path, format=fmt, dpi=PAPER_CONFIG['dpi'],
                       bbox_inches='tight', pad_inches=0.1)
            saved_paths.append(str(file_path))
        
        plt.close(fig)
        
        return saved_paths[0]  # Return primary format path


if __name__ == "__main__":
    # Test the plotting system
    print("Testing Paper Plot Generator...")
    
    # Create plot generator
    plotter = PaperPlotGenerator(output_dir="./test_figures")
    
    # Test model comparison plot
    model_results = {
        'NSRPO': {'accuracy': 0.85, 'perplexity': 2.3, 'kl_divergence': 0.12},
        'GRPO': {'accuracy': 0.82, 'perplexity': 2.6, 'kl_divergence': 0.08},
        'Baseline': {'accuracy': 0.78, 'perplexity': 3.1, 'kl_divergence': 0.15}
    }
    
    path1 = plotter.plot_model_comparison(model_results, save_name="test_comparison")
    print(f"✓ Created model comparison plot: {path1}")
    
    # Test ablation study plot
    ablation_data = [
        {'component_name': 'alpha_1_ce_loss', 'component_value': 0.0, 'metrics': {'accuracy': 0.75}},
        {'component_name': 'alpha_1_ce_loss', 'component_value': 0.1, 'metrics': {'accuracy': 0.85}, 'is_baseline': True},
        {'component_name': 'alpha_1_ce_loss', 'component_value': 0.2, 'metrics': {'accuracy': 0.82}},
    ]
    
    path2 = plotter.plot_ablation_study(ablation_data, 'alpha_1_ce_loss', save_name="test_ablation")
    print(f"✓ Created ablation study plot: {path2}")
    
    # Test heatmap
    matrix = np.random.rand(3, 4)
    path3 = plotter.plot_performance_heatmap(
        matrix, ['NSRPO', 'GRPO', 'Baseline'], 
        ['Acc', 'PPL', 'KL', 'Speed'], save_name="test_heatmap"
    )
    print(f"✓ Created performance heatmap: {path3}")
    
    print("✓ Paper plot generator test completed successfully!")
    print("  Features implemented:")
    print("  - IEEE/ACM standard formatting")
    print("  - Model comparison plots with significance testing")
    print("  - Training curve visualization")
    print("  - Ablation study plots")
    print("  - Performance heatmaps")
    print("  - Statistical comparison plots")
    print("  - Correlation matrices")
    print("  - Multi-panel summary figures")
    print("  - Vector graphics output (PDF, EPS, PNG)")