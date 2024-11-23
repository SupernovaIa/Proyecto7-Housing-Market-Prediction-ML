import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


def plot_numeric_distribution(df, first, last, col, n=1, size = (10, 5), rotation=45):
    # Validate inputs
    if n <= 0:
        raise ValueError("Bin width (n) must be a positive integer.")
    if first >= last:
        raise ValueError("'first' must be less than 'last'.")
    
    # Define the bin edges to align with ticks
    bin_edges = np.arange(first, last + n, n)
    
    # Filter the data
    filtered_data = df[df[col].between(first, last)][col]
    
    if filtered_data.empty:
        print(f"No data available for the range {first} to {last}.")
        return

    # Set dynamic figure size based on the range
    plt.figure(figsize=size)
    
    # Create the histogram with aligned bins
    sns.histplot(filtered_data, bins=bin_edges, kde=False, color="skyblue", edgecolor="black")
    
    # Add title and labels
    plt.title(f"Distribution of {col} ({first}–{last})")
    plt.xlabel("")
    plt.ylabel("Frequency")
    
    # Set x-ticks to align with bins
    plt.xticks(bin_edges, rotation=rotation)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_categoric_distribution(df, col, size = (8, 4), color='mako', rotation=45):

    plt.figure(figsize=size)

    sns.countplot(
        x=col, 
        data=df[col].reset_index(),  
        palette=color, 
        order=df[col].value_counts().index,
        edgecolor="black"
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel("")
    plt.ylabel("Number of registrations")

    plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, size = (5, 5)):

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=size)

    # Mask to make it triangular
    mask = np.triu(np.ones_like(corr_matrix, dtype = np.bool_))
    sns.heatmap(corr_matrix, 
                annot=True, 
                vmin = -1, 
                vmax = 1, 
                mask=mask)
    

def plot_relation_tv_numeric(df, tv, size = (15, 10)):
    
    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_filas = math.ceil(n_plots/2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize = size)
    axes = axes.flat

    for i, col in enumerate(cols_num):

        if col == tv:
            fig.delaxes(axes[i])

        else:
            sns.scatterplot(x = col,
                        y = tv,
                        data = df_num,
                        ax = axes[i], 
                        palette = 'mako')
            
            axes[i].set_title(col)
            axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()



def plot_outliers(df):

    df_num = df.select_dtypes(include = np.number)
    cols_num = df_num.columns

    n_plots = len(cols_num)
    num_rows = math.ceil(n_plots/2)

    cmap = plt.cm.get_cmap('mako', n_plots)
    color_list = [cmap(i) for i in range(cmap.N)]

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize = (9, 5))
    axes = axes.flat

    for i, col in enumerate(cols_num):

        sns.boxplot(x = col, 
                    data = df_num,
                    ax = axes[i],
                    color=color_list[i]) 
        
        axes[i].set_title(f'{col} outliers')
        axes[i].set_xlabel('')

    # Remove last plot, if empty
    if n_plots % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def value_counts(df, col):

    return pd.concat([df[col].value_counts(), df[col].value_counts(normalize=True).round(2)], axis=1)


def quick_plot_numeric(df, col, num=10, size=(10,5), rotation=45):

    max_ = df[col].max()
    min_ = df[col].min()
    n = (max_ - min_) // num

    if n < 2:
        return
    
    plot_numeric_distribution(df, min_, max_, col, n, size=size, rotation=rotation)


def plot_groupby_median(df, groupby, col, max_entries, size=(12, 6), palette='mako'):

    median_by = (
        df.groupby(groupby)[col]
        .median()
        .sort_values()
        .reset_index()
    )

    plt.figure(figsize=size)
    sns.barplot(
        data=median_by.iloc[:max_entries],
        x=col,
        y=groupby,
        palette=palette,
        edgecolor="black"
    )

    plt.title(f'{col} median per {groupby}')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()

    plt.show()