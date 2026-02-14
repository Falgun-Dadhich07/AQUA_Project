################################################################################
######################### VISUALIZATION MODULE ##################################
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def create_var_comparison_plot(test_sets, stock, period, output_dir="plots"):
    """
    Create a comprehensive VaR comparison plot showing:
    - Actual losses
    - VaR forecasts from all models
    - Violation points
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = test_sets[stock].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Plot 1: VaR Forecasts and Losses
    ax1.plot(df['Date'], df['one_day_loss'], label='Actual Loss', 
             color='black', linewidth=1.5, alpha=0.7)
    
    colors = {
        'non_par_VaR': '#1f77b4',
        'CMM_VaR': '#ff7f0e',
        'GARCH_VaR': '#2ca02c',
        'LSTM_MDN_vanilla': '#d62728',
        'LSTM_MDN_reg': '#9467bd',
        'LSTM_MDN_3C': '#8c564b'
    }
    
    labels = {
        'non_par_VaR': 'Historical',
        'CMM_VaR': 'CMM',
        'GARCH_VaR': 'GARCH',
        'LSTM_MDN_vanilla': 'LSTM-MDN Vanilla',
        'LSTM_MDN_reg': 'LSTM-MDN Regularized',
        'LSTM_MDN_3C': 'LSTM-MDN 3-Component'
    }
    
    for var_col in colors.keys():
        if var_col in df.columns and not df[var_col].isna().all():
            ax1.plot(df['Date'], df[var_col], label=labels[var_col], 
                    color=colors[var_col], linewidth=1.2, alpha=0.8, linestyle='--')
    
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Loss / VaR', fontsize=12, fontweight='bold')
    ax1.set_title(f'VaR Forecasts Comparison - {stock} ({period} period)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Violations
    for var_col in colors.keys():
        if var_col in df.columns and not df[var_col].isna().all():
            violations = df[df['one_day_loss'] > df[var_col]]
            if len(violations) > 0:
                ax2.scatter(violations['Date'], [labels[var_col]] * len(violations), 
                           color=colors[var_col], s=100, alpha=0.7, marker='x')
    
    ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('VaR Violations by Model', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{stock}_var_comparison_{period}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def create_backtesting_summary_plot(results_backtesting, stocks, period, output_dir="plots"):
    """
    Create bar plots showing backtesting statistics for all models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['proportion_overshoots', 'LRuc', 'LRcci', 'LRcc']
    titles = ['Proportion of Overshoots', 'Kupiec Test (LRuc)', 
              'Christoffersen Test (LRcci)', 'Combined Test (LRcc)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Aggregate data across all stocks
        data = []
        for i, stock in enumerate(stocks):
            df_result = results_backtesting[i]
            for model in df_result.index:
                value = df_result.loc[model, metric]
                if not pd.isna(value):
                    data.append({'Stock': stock, 'Model': model, 'Value': value})
        
        if len(data) > 0:
            plot_df = pd.DataFrame(data)
            
            # Create bar plot
            models = plot_df['Model'].unique()
            x = np.arange(len(models))
            width = 0.8 / len(stocks) if len(stocks) > 1 else 0.6
            
            for i, stock in enumerate(stocks):
                stock_data = plot_df[plot_df['Stock'] == stock]
                values = [stock_data[stock_data['Model'] == m]['Value'].values[0] 
                         if len(stock_data[stock_data['Model'] == m]) > 0 else 0 
                         for m in models]
                ax.bar(x + i * width, values, width, label=stock, alpha=0.8)
            
            ax.set_xlabel('Model', fontsize=11, fontweight='bold')
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(stocks) - 1) / 2)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add reference lines
            if metric == 'proportion_overshoots':
                ax.axhline(y=0.01, color='red', linestyle='--', linewidth=1, 
                          alpha=0.7, label='Expected (1%)')
            elif metric in ['LRuc', 'LRcci']:
                ax.axhline(y=3.841, color='red', linestyle='--', linewidth=1, 
                          alpha=0.7, label='Critical value (5%, df=1)')
            elif metric == 'LRcc':
                ax.axhline(y=5.991, color='red', linestyle='--', linewidth=1, 
                          alpha=0.7, label='Critical value (5%, df=2)')
            
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Backtesting Results Summary ({period} period)', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'backtesting_summary_{period}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def create_model_performance_heatmap(results_backtesting, stocks, period, output_dir="plots"):
    """
    Create a heatmap showing model performance across stocks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a matrix of LRcc values (combined test statistic)
    models = results_backtesting[0].index.tolist()
    data_matrix = np.zeros((len(stocks), len(models)))
    
    for i, stock in enumerate(stocks):
        for j, model in enumerate(models):
            value = results_backtesting[i].loc[model, 'LRcc']
            data_matrix[i, j] = value if not pd.isna(value) else 0
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(stocks) * 0.5)))
    
    # Create heatmap
    sns.heatmap(data_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                xticklabels=models, yticklabels=stocks, 
                cbar_kws={'label': 'LRcc Statistic'}, ax=ax,
                vmin=0, vmax=10, center=5.991)
    
    ax.set_title(f'Model Performance Heatmap - LRcc Statistic ({period} period)\n' + 
                 'Lower is better | Red line at 5.991 (critical value, 5%, df=2)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stock', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'performance_heatmap_{period}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def create_returns_distribution_plot(stock_data, stock, period, output_dir="plots"):
    """
    Create distribution plots for returns
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = stock_data[stock]
    returns = df['R'].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram with KDE
    ax1 = axes[0]
    ax1.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit normal distribution
    from scipy.stats import norm
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={std:.4f})')
    
    ax1.set_xlabel('Returns', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Returns Distribution - {stock}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    from scipy.stats import probplot
    probplot(returns, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot - {stock}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{stock}_returns_distribution_{period}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def create_all_visualizations(stock_data, test_sets, results_backtesting, stocks, period):
    """
    Create all visualizations
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for stock in stocks:
        print(f"\nGenerating plots for {stock}...")
        
        # VaR comparison plot
        create_var_comparison_plot(test_sets, stock, period, output_dir)
        
        # Returns distribution
        create_returns_distribution_plot(stock_data, stock, period, output_dir)
    
    # Summary plots across all stocks
    print(f"\nGenerating summary plots...")
    create_backtesting_summary_plot(results_backtesting, stocks, period, output_dir)
    create_model_performance_heatmap(results_backtesting, stocks, period, output_dir)
    
    print("\n" + "="*60)
    print(f"✓ All plots saved to '{output_dir}/' directory")
    print("="*60)
