import numpy as np
import rescompy as rc
import rc_helpers as rch
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Union, Callable
from scipy.stats import wasserstein_distance
import functools

def bifurcation_diagram(
        train_library:          rch.Library = None,
        truth_library:          rch.Library = None,
        predictions:            List[rc.PredictResult] = None,
        focus_library:          rch.Library = None,
        focus_colors:           Union[str, List[str], tuple, List[tuple]] = None,
        focus_label:            str = None,
        ghostfocus_library:     rch.Library = None,
        ghostfocus_colors:      Union[str, List[str], tuple, List[tuple]] = None,
        highlight_divergence:   bool = True,
        divergence_color:       Union[str, tuple] = "yellow",
        divergence_alpha:       float = 1,
        pred_discard:           int = 0,
        skip_truth:             bool = False,
        alpha:                  float = 1,
        font_size:              float = 15,
        marker_size:            float = None,
        train_color:            Union[str, tuple] = "black",
        truth_color:            Union[str, tuple] = "tab:blue",
        pred_color:             Union[str, tuple] = "tab:red",
        ymin:                   float = None,
        ymax:                   float = None,
        xlabel:                 str = "Parameter",
        ylabel:                 str = "State Variable",
        figsize:                tuple = None,
        train_linewidth:        float = None,
        plot_train_lines:       bool = False,
        plot_train_points:      bool = False,
        use_legend:             bool = True,
        legend_loc:             str = "best",
        legend_bbox:            tuple = None,
        prediction_method:      str = "Predictions",
        ax:                     mpl.axes._axes.Axes = None,
        remove_xticks:          bool = False,
        remove_yticks:          bool = False,
        train_label:            str = "Training Systems",
        label_truth:            bool = True,
        add_text:               str = None
        ):
    
    with mpl.rc_context({'font.size': font_size}): 
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout = True, figsize = figsize)
        
        ax.set_ylim(ymin, ymax)
        num_markers = 0
        if predictions is not None:
            divergence_counter = 0
            for i, (r, prediction) in enumerate(zip(truth_library.parameters, predictions)):
                if np.all((prediction.reservoir_outputs[pred_discard:] > ymax) |
                          (prediction.reservoir_outputs[pred_discard:] < ymin)):
                    if divergence_counter == 0:
                        label = None
                    else:
                        label = None
                    ax.axvline(x = r, linewidth = train_linewidth,
                               c = divergence_color, alpha = divergence_alpha, label = label)
                    divergence_counter += 1
        if truth_library is not None and not skip_truth:
            for i, (r, trajectory) in enumerate(zip(truth_library.parameters, truth_library.data)):
                if i == 0 and label_truth:
                    label = "Truth"
                    num_markers += 1
                else:
                    label = None
                ax.scatter(r * np.ones(trajectory.shape), trajectory,
                           s = marker_size, c = truth_color, alpha = alpha,
                           label = label)
        if predictions is not None and truth_library is not None:
            for i, (r, prediction) in enumerate(zip(truth_library.parameters, predictions)):
                if i == 0 and prediction_method is not None:
                    label = prediction_method
                    num_markers += 1
                else:
                    label = None
                ax.scatter(r * np.ones(prediction.reservoir_outputs[pred_discard:].shape),
                           prediction.reservoir_outputs[pred_discard:],
                           s = marker_size, c = pred_color, alpha = alpha,
                           label = label)
        if train_library is not None:
            if plot_train_lines:
                for i, r in enumerate(np.array(train_library.parameters).flatten()):
                    if i == 0:
                        label = train_label
                    else:
                        label = None
                    print(r)
                    ax.axvline(r, ymax = .1, c = train_color, label = label, #alpha = alpha
                               linewidth = train_linewidth, marker = "^")
            elif plot_train_points:
                for i, (r, trajectory) in enumerate(zip(train_library.parameters, train_library.data)):
                    if i == 0:
                        label = train_label
                        num_markers += 1
                    else:
                        label = None
                    ax.scatter(r * np.ones(trajectory.shape), trajectory,
                               s = marker_size, c = train_color, alpha = alpha,
                               label = label)
        if ghostfocus_library is not None:
            for i, r in enumerate(np.array(ghostfocus_library.parameters).flatten()):
                num_markers += 1
                label = ghostfocus_library.parameter_labels[0] + f"$_{i+1}^*={r}$"
                ax.axvline(r, ymax = .05, c = ghostfocus_colors[i], label = label, #alpha = alpha
                           linewidth = 0, marker = "none", fillstyle = "full")
        if focus_library is not None:
            for i, r in enumerate(np.array(focus_library.parameters).flatten()):
                num_markers += 1
                label = focus_label + f"$_{i+1}^*={r}$"
                ax.axvline(r, ymax = .05, c = focus_colors[i], label = label, #alpha = alpha
                           linewidth = train_linewidth, marker = "^", fillstyle = "full")
                
        if truth_library is not None and train_library is not None:
            ax.set_xlim(min(min(truth_library.parameters), min(train_library.parameters)),
                        max(max(truth_library.parameters), max(train_library.parameters)))
        elif truth_library is not None:
            ax.set_xlim(min(truth_library.parameters), max(truth_library.parameters))
        elif train_library is not None:
            ax.set_xlim(min(train_library.parameters), max(train_library.parameters))
        if use_legend:
            legend = ax.legend(frameon = False, loc = legend_loc, bbox_to_anchor = legend_bbox)
            for leg_i in range(num_markers):
                legend.legendHandles[leg_i]._sizes = [30]
            if plot_train_lines and train_label is not None:
                if (focus_library is None and ghostfocus_library is None):
                    legend.legendHandles[-1].set_linestyle("none")
                elif focus_library is not None and ghostfocus_library is None:
                    for j in range(1, len(focus_library.parameters) + 2):
                        legend.legendHandles[-j].set_linestyle("none")
                elif ghostfocus_library is not None and focus_library is None:
                    for j in range(1, len(ghostfocus_library.parameters) + 2):
                        legend.legendHandles[-j].set_linestyle("none")
                elif focus_library is not None and ghostfocus_library is not None:
                    for j in range(1, len(focus_library.parameters) + len(ghostfocus_library.parameters) + 2):
                        legend.legendHandles[-j].set_linestyle("none")
                        legend.legendHandles[-j].set_marker("^")
        
        if remove_xticks:
            ax.set_xticks([])
        else:
            ax.set_xlabel(xlabel)
        if remove_yticks:
            ax.set_yticks([])            
        else:
            ax.set_ylabel(ylabel)
        if add_text is not None:
            ax.text(.01, .99, add_text, ha = 'left', va = 'top',
                    weight = "bold",
                    transform = ax.transAxes)
        
def get_wasserstein(
        predictions:        List[rc.PredictResult],
        discard:            int = 0,
        normalize:          bool = False
        ):
    
    '''
    Note: Adapt to multi-dimensional case in some reasonable way.
    '''
    
    if normalize:    
        return [2 * wasserstein_distance(
            prediction.reservoir_outputs[discard:].flatten(),
            prediction.target_outputs[discard:].flatten())
            / np.ptp(prediction.target_outputs)
            for prediction in predictions]
    else:
        return [wasserstein_distance(
            prediction.reservoir_outputs[discard:].flatten(),
            prediction.target_outputs[discard:].flatten())
            for prediction in predictions]

def cumulative_probabilities(
        data:               np.ndarray,
        xs:                 Union[list, np.ndarray] = np.linspace(0, 1, 100)
        ):
    
    return [np.where(data.flatten() < x)[0].shape[0] /
            data.flatten().shape[0] for x in xs]

def plot_cumulative_probabilities(
        predictions:            List[rc.PredictResult],
        truth_library:          rch.Library,
        interest_parameters:    List[float],
        truth_colors:           Union[List[str], List[tuple]] = None,
        predicted_colors:       Union[List[str], List[tuple]] = None,
        pred_discard:           int = 0,
        parameter_label:        str = "p",
        val_min:                float = None,
        val_max:                float = None,
        num_val_samples:        float = None,
        font_size:              float = 15.,
        xlabel:                 str = "State Variable",
        ylabel:                 str = "Cumulative Probability",
        make_ax_legend:         bool = False,
        legend_loc:             str = "best",
        legend_bbox:            tuple = None,
        truth_marker:           str = "o",
        pred_marker:            str = "*",
        truth_linestyle:        str = "-",
        pred_linestyle:         str = "--",
        linewidth:              float = None,
        marker_size:            float = None,
        ax:                     mpl.axes._axes.Axes = None,
        remove_xticks:          bool = False,
        remove_yticks:          bool = False,
        add_text:               str = None,
        alpha:                  float = None
        ):
    
    if val_min is None:
        val_min = min([np.min(p.reservoir_outputs) for p in predictions])
    if val_max is None:
        val_max = max([np.max(p.reservoir_outputs) for p in predictions])
    
    with mpl.rc_context({'font.size': font_size}):
        vals = np.linspace(val_min, val_max, num_val_samples)
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout = True)
        for ri, r in enumerate(interest_parameters):
            nearest_pred = np.argmin(np.abs(np.array(truth_library.parameters)[:,0] - r))
            print("Closest Match: ", np.array(truth_library.parameters)[nearest_pred, 0])
            prediction = predictions[nearest_pred].reservoir_outputs[pred_discard:].flatten()
            truth = predictions[nearest_pred].target_outputs[pred_discard:].flatten()
            ax.plot(vals, cumulative_probabilities(prediction, vals),
                    label = parameter_label + f"$^*_{ri+1}$" + ", Predicted",
                    linewidth = linewidth,
                    linestyle = pred_linestyle, ms = marker_size,
                    marker = pred_marker,
                    color = predicted_colors[ri],
                    alpha = alpha)
            ax.plot(vals, cumulative_probabilities(truth, vals),
                    label = parameter_label + f"$^*_{ri+1}$" +  ", Truth", 
                    linewidth = linewidth,
                    linestyle = truth_linestyle,  ms = marker_size,
                    marker = truth_marker,
                    color = truth_colors[ri],
                    alpha = alpha)
        
        ax.set_xlim(val_min, val_max)
        ax.set_ylim(0, 1)  
        
        if remove_xticks:
            ax.set_xticks([])
        else:
            ax.set_xlabel(xlabel)
        if remove_yticks:
            ax.set_yticks([])
        else:
            ax.set_ylabel(ylabel)
        if add_text is not None:
            ax.text(.01, .99, add_text, ha = 'left', va = 'top',
                    weight = "bold", transform = ax.transAxes)
        if make_ax_legend:
            leg = ax.legend(loc = legend_loc, frameon = False, bbox_to_anchor = legend_bbox)
        
def get_map_error(
        predictions:        Union[rc.PredictResult, List[rc.PredictResult]],
        analytic_map:       Callable,
        discard:            int = 0,
        normalize:          bool = True
        ):
    
    if isinstance(predictions, rc.rescompy.PredictResult):
        predictions = [predictions]
    
    if normalize:
        norm = np.mean([np.sqrt(np.sum(np.square(
            prediction.target_outputs[discard + 1:] - prediction.target_outputs[discard: -1]
            ), axis = 1)) for prediction in predictions])
    else:
        norm = 1
    
    try:
        errors = [prediction.reservoir_outputs[discard + 1:] -
                  analytic_map(prediction.reservoir_outputs[discard: -1])
                  for prediction in predictions]
    except:
        errors = [prediction.reservoir_outputs[discard + 1:] - np.array([
                  analytic_map(prediction_step[None]).flatten() for prediction_step in
                  list(prediction.reservoir_outputs[discard: -1])])
                  for prediction in predictions]
    
    if len(errors) > 1:
        errors = functools.reduce(
            lambda x, y : np.concatenate((x, y), axis = 0),
            errors, errors[0])
    else:
        errors = errors[0]
        
    map_error_mean = np.mean(np.sqrt(np.sum(np.square(errors), axis = 1))) / norm
    map_error_stderr = np.std(np.sqrt(np.sum(np.square(errors), axis = 1))) / (
        norm * np.sqrt(errors.shape[0]))
    
    return map_error_mean, map_error_stderr

def get_nstep_error(
        predictions:        Union[rc.PredictResult, List[rc.PredictResult]],
        nsteps:             int = 10,
        normalize:          bool = True
        ):
    
    if isinstance(predictions, rc.rescompy.PredictResult):
        predictions = [predictions]
    
    if normalize:
        errors = [prediction.nrmse[nsteps].mean() for prediction in predictions]
    else:
        errors = [prediction.rmse[nsteps].mean() for prediction in predictions]
    
    if len(errors) > 1:
        errors = functools.reduce(
            lambda x, y : np.concatenate((x, y), axis = 0),
            errors, errors[0])
    else:
        errors = errors[0]
        
    map_error_mean = np.mean(errors)
    map_error_stderr = np.std(errors)
    
    return map_error_mean, map_error_stderr

def plot_one_step_map(
        prediction:         rc.PredictResult,
        analytic_xs:        np.ndarray = None,
        analytic_map:       Callable = None,
        discard:            int = 0,
        font_size:          float = 15,
        xlabel:             str = "$x_n$",
        ylabel:             str = "$x_{n+1}$",
        title:              str = None,
        analytic_color:     Union[str, tuple] = "grey",
        truth_color:        Union[str, tuple] = "tab:blue",
        pred_color:         Union[str, tuple] = "tab:red",
        truth_linewidth:    float = None,
        scatter_truth:      bool = False,
        plot_truth:         bool = True,
        xmin:               float = None,
        xmax:               float = None,
        ymin:               float = None,
        ymax:               float = None,
        ax:                 mpl.axes._axes.Axes = None,
        marker_size:        float = None,
        add_text:           str = None,
        label_lines:        bool = True,
        alpha:              float = None,
        make_ax_legend:     bool = False,
        legend_loc:         str = "best",
        legend_bbox:        tuple = None,
        parameter_label:    str = None
        ):
    
    if analytic_xs is None:
        truth = np.sort(prediction.target_outputs[discard:].flatten())
        truth_diffs = np.abs(np.diff(truth))
        breaks_start = np.argwhere(truth_diffs > 50 * np.mean(truth_diffs)) + 1
        breaks_end = np.argwhere(truth_diffs > 50 * np.mean(truth_diffs))
        if len(breaks_start) > 0:
            breaks_start = [0] + list(breaks_start[0])
        else:
            breaks_start = [0]
        if len(breaks_end) > 0:
            breaks_end = list(breaks_end[0]) + [truth.shape[0] - 1]
        else:
            breaks_end = [truth.shape[0] - 1]
        analytic_xs = [np.linspace(truth[breaks_start[i]], truth[breaks_end[i]], 100)
                       for i in range(len(breaks_start))]
    
    with mpl.rc_context({'font.size': font_size}): 
        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout = True)
        if label_lines and parameter_label is None:
            labels = ["Analytical", "Truth", "Predictions"]
        elif label_lines:
            labels = [parameter_label + ", Truth",
                      parameter_label + ", Truth",
                      parameter_label + ", Predicted"]
        else:
            labels = [None, None, None]
        if plot_truth and (analytic_xs is not None and analytic_map is not None):
            for xi, xs in enumerate(analytic_xs):
                if xi == 0:
                    a_label = labels[0]
                else:
                    a_label = None
                    
                ax.plot(xs, analytic_map(xs),
                        linestyle = "-", c = truth_color,
                        linewidth = truth_linewidth,
                        label = a_label
                        )
        if scatter_truth:
            ax.scatter(prediction.target_outputs[discard: -1],
                       prediction.target_outputs[discard + 1:],
                       c = truth_color, s = marker_size,
                       label = labels[1], alpha = alpha
                       )
        ax.scatter(
            prediction.reservoir_outputs[discard: -1],
            prediction.reservoir_outputs[discard + 1:],
            c = pred_color, s = marker_size,
            label = labels[2],
            alpha = alpha
            )
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        if not label_lines:
            ax.legend(frameon = False, loc = "best")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if add_text is not None:
            ax.text(.01, .99, add_text, ha = 'left', va = 'top',
                    weight = "bold", transform = ax.transAxes)
            
        if make_ax_legend:
            leg = ax.legend(loc = legend_loc, frameon = False, bbox_to_anchor = legend_bbox)
            for leg_i in range(len(leg.legendHandles)):
                leg.legendHandles[leg_i].set_alpha(1)