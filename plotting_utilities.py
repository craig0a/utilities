import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



from scipy.stats import gaussian_kde
def feature_distributions(data, features, value_col, ncols = 3, 
                          highlighting = None, edge_values = None,
                          xmin = None, xmax = None, n_bins = 100, n_grid = 500, 
                          min_group_size = 100, max_group_spike = 1.2, log_y = True, 
                          w_scale = 5, h_scale = 5, pad = 0.4, w_pad = 2.0, h_pad = 2.0, 
                          anchor_legend = None):
    
    """
    Plot continuous feature distributions as split by secondary category. 
    
    If there are enough sample of a type in the group of secondary category, the KDE will be plotted. Otherwise, the histogram will be plotted. 
    
    Options to add categorical highlights binning the continuous value (not recommended if plotting histograms) and/or marked edge labels. 

    Arguments:
        data (pandas dataframe): pandas dataframe
        features (list of strings): list of columns in date to plot (each as separate axis), split based on values in column
        value_col (string): column name for continuous variable against which to plot
        xmin (float): minimum value to plot
        xmax (float): maximum value to plot
        n_bins (int): number of bins for histogram
        n_grid (int): resolution of KDE
        highlighting (list of tuples: (float, float, color, string)): lower bound, upper bound, color, and label for highlighted ranges of continuous variable 
        edge_values (list of tuples: (float, color, string)): x-location, color, and label for a vertical line marking a specific value of the continuous variable 
        n_cols (int): number of columns for figure
        min_group_size (int): number of samples for a given group in order to plot as KDE
        max_group_spike (float): if KDE is unreasonably spiked (i.e. no distribution), plot histogram instead
        log_y (bool): set y-scale to be logarithmic
        w_scale (float): aspect ratio of axes
        h_scale (float): aspect ratio of axes
        pad (float): padding of subplots
        w_pad (float): padding of subplots
        h_pad(float): padding of subplots
        anchor_legend (tuple of floats): x,y coordinates to pin legend to axis
       

    Returns:
        fig, ax: figure and axis handles
    """


    nrows = int(np.ceil(len(features)/float(ncols)))

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, 
                           sharey = False, sharex = False, 
                           figsize = (w_scale*ncols,h_scale*nrows))
    ax= ax.reshape(-1)
    fig.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)

    for iax, f in enumerate(features):
        feature_data = data[[f, value_col]].dropna(subset = [value_col])

        if xmin is None:
            xmin = feature_data[value_col].min()
        if xmax is None:
            xmax = feature_data[value_col].max()

        kde_x = np.linspace(xmin,xmax,n_grid)
        bin_width = (xmax-xmin)/n_bins
        bins_x = np.arange(xmin-bin_width,xmax+bin_width, bin_width)+bin_width/2.

        grouped_feature = feature_data.groupby(f)

        for n, g in grouped_feature:
            g = g[value_col]
            if len(g.unique())>1:
                kde_y = gaussian_kde(g)(kde_x)
                if (np.max(kde_y) < max_group_spike)&(g.size > min_group_size):
                    ax[iax].plot(kde_x, kde_y, label = '%s: %d' %(n, g.size)) 
                else:
                    g.plot.hist(ax = ax[iax], bins = bins_x, align = 'mid', alpha = 0.25, 
                                density=True, label = '%s: %d' %(n, g.size))
            else:
                g.plot.hist(ax = ax[iax], bins = bins_x, align = 'mid', alpha = 0.25, 
                            density=True, label = '%s: %d' %(n, g.size))

        ax[iax].set_xlabel(value_col)
        if log_y:
            ax[iax].set_yscale('log')
            ax[iax].set_ylabel('log PDF')
        else:
            ax[iax].set_ylabel('PDF')
        
        if highlighting is not None:
            for lb, rb, c, label in highlighting:
                ax[iax].axvspan(lb, rb, alpha=0.25, color=c, label = label)
            
        if edge_values is not None:
            for v, c, label in edge_values:
                ax[iax].axvline(x=v, color=c, linewidth = 3., 
                        linestyle = '--',  label = label)
                
        ax[iax].legend(title = f)
        if anchor_legend is not None:
            ax[iax].legend(bbox_to_anchor= anchor_legend)
        
    return fig, ax

