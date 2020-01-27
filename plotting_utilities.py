import matplotlib
from matplotlib.gridspec import GridSpec
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



def jointplot(x, y, c = 'k', cmap = 'gray_r',
              xmin = None, xmax = None, xdelta = None,  
              ymin = None, ymax = None, ydelta = None, 
              logscale = False, gridsize = 50, bins = None, alpha = 0.2,
              joint_xlabel = None, joint_ylabel = None,
              marginal_xlabel = None, marginal_ylabel = None, 
              fig_axes = None, joint_type = 'hex', scatter_label = '',
              highlighting = None, edge_values = None, anchor_legend = None):
    """
     Joint plot continuous feature distributions. 
     Option to plot as hexbinned or scatter joint distribution
     Option to add categorical highlights binning the continuous value (not recommended if plotting histograms) and/or marked edge labels. 

    Arguments:
        x (pandas series, floats): x-axis continuous values
        y (pandas series, floats): y-axis continuous values
        c (color identifier):
        c_map (colormap identifier):
        alpha: 
        xmin (float): minimum value to plot on x axis 
        xmax (float): maximum value to plot on x axis
        xdelta
        ymin (float): minimum value to plot on y axis 
        ymax (float): maximum value to plot on y axis
        ydelta
        bins (int): number of bins for histogram
        gridsize (int): resolution of hexbin
        highlighting (list of tuples: (float, float, color, string)): lower bound, upper bound, color, and label for highlighted ranges of continuous variable 
        edge_values (list of tuples: (float, color, string)): x-location, color, and label for a vertical line marking a specific value of the continuous variable 
        logscale (bool): set y-scale to be logarithmic
        anchor_legend (tuple of floats): x,y coordinates to pin legend to axis
        fig_axes

    Returns:
        fig, ax_joint, ax_marg_x, ax_marg_y : handles to figure and each of the three subplot axes
    """

    if fig_axes == None:
        fig = plt.figure()
        gs = GridSpec(4,4)

        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_x = fig.add_subplot(gs[0,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])
    else:
        fig,ax_joint,ax_marg_x,ax_marg_y = fig_axes
       
    if joint_type == 'hex':
        ax_joint.hexbin(x,y, cmap = cmap, bins= 'log', gridsize = gridsize )
    elif joint_type == 'scatter':
        ax_joint.scatter(x,y, color = c, alpha= alpha, label = scatter_label)
        
   
    if xmin is None:
        xmin = min(x)
    if xmax is None:
        xmax = max(x)
    if ymin is None:
        ymin = min(y)
    if ymax is None:
        ymax = max(y)
        
    if bins:
        ax_marg_x.hist(x, density = False, color = c, alpha = alpha, bins = bins[0], 
                       align = 'mid')
        ax_marg_y.hist(y, density = False, color = c, alpha = alpha, bins = bins[1], 
                       align = 'mid', orientation="horizontal")
    else:  
        ax_marg_x.hist(x, density = False, color = c, alpha = alpha, range = (xmin, xmax), 
                       align = 'mid')
        ax_marg_y.hist(y, density = False, color = c, alpha = alpha,  range = (ymin, ymax), 
                       align = 'mid', orientation="horizontal")
    
    if logscale:
        ax_joint.set_xscale('log')
        ax_joint.set_yscale('log')
        ax_marg_x.set_xscale('log')
        ax_marg_x.set_yscale('log')
        ax_marg_y.set_xscale('log')
        ax_marg_y.set_yscale('log')
    else:
        if xdelta is None:
            xdelta = (xmax - xmin)/100.
        if ydelta is None:
            ydelta = (ymax - ymin)/100.
        ax_joint.axis([xmin-xdelta, xmax+xdelta, ymin-ydelta, ymax+ydelta])
        ax_marg_x.set_xlim([xmin-xdelta, xmax+xdelta])
        ax_marg_y.set_ylim([ymin-ydelta, ymax+ydelta])

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    if joint_xlabel is None:
        try:
            joint_xlabel = x.name
        except:
            joint_xlabel = ''
    if joint_ylabel is None:
        try:
            joint_ylabel = y.name
        except:
            joint_ylabel = ''
        
    ax_joint.set_xlabel(joint_xlabel)
    ax_joint.set_ylabel(joint_ylabel)

    # Set labels on marginals
    if marginal_xlabel is None:
        marginal_xlabel = 'Count'
    if marginal_ylabel is None:
        marginal_ylabel = 'Count'
        
    ax_marg_y.set_xlabel(marginal_xlabel)
    ax_marg_x.set_ylabel(marginal_ylabel )
    
    if highlighting is not None:
        for lb, rb, c, label in highlighting:
            ax_joint.axvspan(lb, rb, alpha=0.25, color=c, label = label)

    if edge_values is not None:
        for v, c, label in edge_values:
            ax_joint.axvline(x=v, color=c, linewidth = 3., 
                    linestyle = '--',  label = label)

    if anchor_legend is not None:
        ax_joint.legend(bbox_to_anchor= anchor_legend)


    return fig, ax_joint, ax_marg_x, ax_marg_y