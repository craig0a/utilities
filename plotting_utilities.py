import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
import sys

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix

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


# Adapted from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels,
            row_title = '', col_title = '', ax=None,
            cbar_kw={}, cbarlabel="", vmin=None, vmax=None,
            x_tick_rotation = 0, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    Returns:
        im, cbar : handles to image and colorbar

    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()
        
    if not vmin:
        vmin = np.amin(data)
        
    if not vmax:
        vmax = np.amax(data)

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=True, left=True)

    ax.set_xlabel(col_title)
    ax.set_ylabel(row_title)

    return im, cbar

# Adapted from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.
    Returns:
        texts: list of text annotations

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_classifier_performance(estimator, metrics_args, f_train, r_train, f_dev, r_dev, 
                                calibrated = True, thresholds = None):
    if calibrated:
        model = CalibratedClassifierCV(estimator, method="sigmoid")
        title = model.__class__.__name__+ ', '+model.base_estimator.__class__.__name__
    else:
        model = estimator
        title = model.base_estimator.__class__.__name__
        
    model.fit(f_train, r_train)
    mr_dev_probs = model.predict_proba(f_dev)
    mr_dev_probs = pd.DataFrame(data = mr_dev_probs, 
                                            columns = model.classes_, 
                                            index = f_dev.index)
    
    if len(model.classes_) == 2:
        c1 = model.classes_[0]
        c2 = model.classes_[1]
        
        ncols = 2
        nrows = 1
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey = False, sharex = False, 
                               figsize = (4*ncols,4*nrows))
        fig.tight_layout(pad=0.4, w_pad=10.0, h_pad=0.0)

        # Find appropriate threshold
        if thresholds is None:
            thresholds = np.linspace(0, 1., 100)
        evaluated_metrics = {}
        for metric, kwargs in metrics_args:
            evaluated_metrics[metric.__name__] = []
        for threshold in thresholds:
            mr_dev_type = mr_dev_probs.idxmax(axis = 1)
            mr_dev_type.loc[(mr_dev_probs[c1] >= threshold)] = c1
                
            for metric, kwargs in metrics_args:
                try:
                    evaluated_metrics[metric.__name__] += [metric(r_dev, mr_dev_type, **kwargs)]
                except:
                    evaluated_metrics[metric.__name__] += [np.nan]
                    continue
                
        if 'loss' in metrics_args[0][0].__name__:
            i = np.argmin(evaluated_metrics[metrics_args[0][0].__name__])
        else:
            i = np.argmax(evaluated_metrics[metrics_args[0][0].__name__])
        
        # Plot first two metrics
        m1 = metrics_args[0][0].__name__
        color = 'r'
        ax[0].set_xlabel('Threshold')
        ax[0].set_ylabel(m1, color=color)
        ax[0].plot(thresholds, evaluated_metrics[m1], color=color, label = m1)
        ax[0].tick_params(axis='y', labelcolor=color)
        if len(metrics_args) > 1:
            ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
            m2 = metrics_args[1][0].__name__
            color = 'b'
            ax2.set_ylabel(m2, color=color)  # we already handled the x-label with ax1
            ax2.plot(thresholds, evaluated_metrics[m2], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        plt.axvline(thresholds[i], color = 'k')
        
        # Additional model information
        model_size = float(sys.getsizeof(pickle.dumps(model)))/1.e6
        title += '\nThreshold: %.2f'%(thresholds[i])+\
                 '\nModel size: %.2f MB'%(model_size)
        for k, v in evaluated_metrics.items():
            title += '\n%s: %.3f'%(k,v[i])
        fig.suptitle(title, y = 1.4)

    else:
        ncols = 1
        nrows = 1
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey = False, sharex = False, 
                               figsize = (8*ncols,8*nrows))
        fig.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
        ax = [ax]
        
        mr_dev_type = mr_dev_probs.idxmax(axis = 1)
        evaluated_metrics = {}
        for metric, kwargs in metrics_args:
            try:
                evaluated_metrics[metric.__name__] = metric(r_dev, mr_dev_type, **kwargs)
            except:
                continue
        
        
        # Additional model information
        model_size = float(sys.getsizeof(pickle.dumps(model)))/1.e6
        title += '\nModel size: %.2f MB'%(model_size)
        for k, v in evaluated_metrics.items():
                title += '\n%s: %.3f'%(k,v)
        fig.suptitle(title, y = 1.1)


    # Plot confusion matrix, labeling with other evaluation metrics
    cm = confusion_matrix(r_dev, mr_dev_type)
    im, cbar = heatmap(cm, 
                       row_title ='True class', row_labels=  model.classes_, 
                       col_title = 'Predicted class', col_labels =  model.classes_,
                       ax=ax[-1], cmap='gray_r', cbarlabel="Counts")
    ax[-1].invert_yaxis()
    ax[-1].invert_xaxis()
    cbar.remove()
    texts = annotate_heatmap(im, valfmt="{x:.0f}")
    normed_cm = cm/float(len(f_dev))*100
    pct_texts = annotate_heatmap(im, data = normed_cm, valfmt="\n\n({x:.3f}%)")


    return mr_dev_type, mr_dev_probs, fig, ax