#%% Import Statements

import rescompy as rc
import rescompy.regressions as regressions
import rescompy.features as features
import numpy as np
from typing import Union, Callable, List
import logging
from dataclasses import dataclass
import os
import pickle
import shutil
import inspect
import numba
import matplotlib.pyplot as plt
import itertools
import functools
import matplotlib as mpl

windower = np.lib.stride_tricks.sliding_window_view


#%% Feature Functions

@dataclass
class States_Polynomial(features.StandardFeature):
    """The Polynomial Feature-getting function.
    
    Returns feature function that returns a concatenation of
    [r, r^2, ..., r^degree].
    
    Args:
        degree (int): The maximum degree of the polynomial.
        
    Returns:
        s (np.ndarray): The feature vectors.
		
	Each instance of this feature function constructed will require jit 
	compilation. If repeated calls are required, it is better to create a 
	single feature object and call it repeatedly within a loop than to create
	a new instance at each iteration.
    """

    degree: int = 2
    
    def __call__(self, r, u):
        s = r
        for poly_ind in range(2, self.degree+1):
            s = np.concatenate((s, r**poly_ind), axis=1)
        return s
	
    def __post_init__ (self):
        degree = self.degree
        
        def inner(r : np.ndarray, u : np.ndarray):
            s = r #np.hstack((const, u, r))
            for poly_ind in range(2, degree+1):
	            s = np.concatenate((s, r**poly_ind), axis=1)
            return s

        self.compiled = numba.njit(inner)

    def feature_size(self, esn_size:int, input_dim:int): 
        return esn_size * self.degree

    def jacobian(r: np.ndarray, u: np.ndarray, dr_du: np.ndarray):
        raise NotImplementedError()

#%% Mapper Functions

@numba.jit(nopython = True, fastmath = True)
def drive_mapper(inputs, outputs):
    return inputs

def get_open_loop_mapper(
		output_dim:	int = 1
		):
	
	"""
	Returns a mapper function for repeated open loop prediction.
	
	Should be called at the top of codes to ensure numba compilation is not 
	repeated unnecessarily.
	"""
	
	@numba.jit(nopython = True, fastmath = True)
	def mapper(		
		inputs: np.ndarray,
		outputs: np.ndarray,
		):
		
		return outputs[:output_dim]
		
	return mapper

#%% Data Storage

@dataclass
class Run_Result:
    
    run_label:                  str
    experimental_parameters:    dict = None
    optimization_info:          dict = None
    prediction_methods:         list = None
    predictions:                dict = None
    feature_functions:          dict = None
    pred_regularizations:       dict = None
    map_regularizations:        dict = None
	
    def save(self, save_loc, safe_save = False, file_name = None):
		
        """        
		Saves the run information in a provided directory.
		
		Args:
			save_loc (str): The absolute or relative path to the folder.
            file_name (str): The name of the file in which the Run_Result will
                             be stored. If None, defaults to "run_data.pickle".
			safe_save (bool): If False, will overwrite existing files and
                              folders.
                              Otherwise, will raise an exception if saving
                              would overwrite anything.
		"""
                
        if file_name is None:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            if os.path.isdir(save_loc):
                if safe_save:
                    msg = f"Already folder or file at '{save_loc}' and " \
                        "safe_save is True."
                    logging.error(msg)
                    raise FileExistsError(msg)
                else:
                    shutil.rmtree(save_loc)
                    msg = f"Already a folder or file at '{save_loc}' but " \
                        "safe_save is False; deleting the existing " \
                        "files and folders."
                    logging.info(msg)
                    
            os.makedirs(save_loc)
            with open(os.path.join(save_loc, "run_data.pickle"), 'wb') as temp_file:
                pickle.dump(self, temp_file)
                
        else:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            head, tail = os.path.split(os.path.join(save_loc, file_name))
            if os.path.isdir(head):
                if os.path.exists(os.path.join(save_loc, file_name + ".pickle")):
                    if safe_save:
                        msg = f"Already folder or file at '{save_loc}' and " \
                            "safe_save is True."
                        logging.error(msg)
                        raise FileExistsError(msg)
                    else:
                        msg = f"Already a folder or file at '{save_loc}' but " \
                              "safe_save is False; deleting the existing " \
                              "files and folders."
                        logging.info(msg)
                        
            else:
                os.makedirs(head)
            
            with open(os.path.join(save_loc, file_name + ".pickle"), 'wb') as temp_file:
                pickle.dump(self, temp_file)

def reduce_prediction(
		predict_result:		rc.PredictResult
		):
	
	"""
	Erases the reservoir states and resync states from a predict_result to 
	reduce the required memory.
	"""
	
	predict_result.resync_states = None
	predict_result.reservoir_states = None
	
	return predict_result

def reduce_train_result(
		train_result:         rc.TrainResult
		):
    
    """
	Erases the reservoir states from train_result to reduce the required memory.
	"""
    
    train_result.states = None
    train_result.listed_states = None
    
    return train_result

#%% Plotting

def plot_spatiotemporal(
        time_series:        np.ndarray,
        colorbar_label:     str = "$u(x, t)$",
        title:              str = None,
        font_size:          float = 15.,
        time_step:          float = None,
        periodicity_length: float = None,
        lyapunov_time:      float = None,
        colormap:           str = None,
        vmin:               float = None,
        vmax:               float = None
        ):
    
    """
    A function to plot trajectories of spatio-temporal systems with one spatial
    dimension.
    """
    
    time_series = time_series.T    
    with mpl.rc_context({"font.size" : font_size}):
        fig, ax = plt.subplots(constrained_layout = True)
        
        if title is not None:
            ax.set_title(title)
        
        x = np.arange(time_series.shape[1])
        y = np.arange(time_series.shape[0])
        
        if lyapunov_time is not None and time_step is not None:
            x = x * time_step / lyapunov_time
            x_label = "$t$ ($\\tau_{Lyap}$)"
        elif time_step is not None:
            x = x * time_step
            x_label = "$t$"
        else:
            x_label = "$t$ ($\\Delta t$)"
            
        if periodicity_length is not None:
            y = y * periodicity_length / time_series.shape[0]
            y_label = "$x$"
        else:
            y_label = "$x$ ($\\Delta x$)"
            
        x, y = np.meshgrid(x, y)
        pcm = ax.pcolormesh(x, y, time_series, shading = "nearest", cmap = colormap,
                            vmin = vmin, vmax = vmax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(pcm, ax = ax, label = colorbar_label)
        plt.show()

def plot_predict_result(
        prediction:         rc.PredictResult,
        plot_dims:          Union[int, List[int]] = None,
        max_horizon:        int = None,
        frame_legend:       bool = False,
        legend_loc:         tuple = (.5, 1.1),
        legend_ax:          int = 0,
        n_legend_cols:      int = 4,
        font_size:          float = 15.,
        incl_tvalid:        bool = True,
        fig:                mpl.figure.Figure = None,
        axes:               np.ndarray = None,
        xlabel:             str = None,
        ylabel:             str = None,
        prediction_color:   str = "r",
        truth_color:        str = "k",
        linewidth:          float = None,
        xlims:              tuple = None,
        ylims:              tuple = None,
        figsize:            tuple = None,
        num_ticks:          int = None,
        line_alpha:         float = 1,
        fig_alpha:          float = 1,
        vert_linewidth:     float = None,
        t0:                 float = 0.
        ):
    
    if plot_dims is None:
        plot_dims = list(np.arange(prediction.reservoir_outputs.shape[1]))
    elif isinstance(plot_dims, int):
        plot_dims = [plot_dims]
        
    if max_horizon is None:
        max_horizon = prediction.reservoir_outputs.shape[0]
        
    if xlabel is None:
        xlabel = "Time (Time Steps, $\\Delta t$)"
    
    with mpl.rc_context({"font.size" : font_size}):
        if figsize is None:
            figsize = (12, 3 * len(plot_dims))
        if axes is None:
            fig, axs = plt.subplots(len(plot_dims), 1,
                                    figsize = figsize, 
                                    sharex = True, constrained_layout = True)
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
        else:
            axs = axes
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
            for ax in axs[1:]:
                ax.sharex(axs[0])
            for ax in axs[:-1]:
                ax.tick_params(labelbottom = False)
        for i, ax in enumerate(axs):
            if prediction.resync_inputs is not None:
                ax.plot(
                    t0 + np.arange(- prediction.resync_inputs.shape[0] + 1, 1),
                    prediction.resync_inputs[:, i],
                    color = truth_color, #"k",
                    #label = "True Signal"
                    linewidth = linewidth,
                    alpha = line_alpha
                    )
            if prediction.resync_outputs is not None:
                lookback_length = prediction.resync_inputs.shape[0] - prediction.resync_outputs.shape[0]
                ax.plot(
                    t0 + np.arange(- prediction.resync_outputs.shape[0] + 1, 1),
                    prediction.resync_inputs[lookback_length:, i],
                    color = truth_color, #"r",
                    linestyle = "-", #"dotted",
                    #label = "True Signal"
                    linewidth = linewidth,
                    alpha = line_alpha
                    )
            if prediction.target_outputs is not None:
                ax.plot(
                    t0 + np.arange(1, prediction.target_outputs.shape[0] + 1),
                    prediction.target_outputs[:, i],
                    color = truth_color, #"k",
                    label = "Truth", #"e Signal"
                    linewidth = linewidth,
                    alpha = line_alpha
                    )
            ax.plot(
                t0 + np.arange(1, prediction.reservoir_outputs.shape[0] + 1),
                prediction.reservoir_outputs[:, i],
                color = prediction_color, #"r",
                label = "Prediction",
                linestyle = "-", #dotted"
                linewidth = linewidth,
                alpha = line_alpha
                )
            ax.axvline(x = t0, linestyle = "--", color = "k",
                       label = "Loop Closed", linewidth = vert_linewidth)
            if prediction.target_outputs is not None and incl_tvalid and \
                np.all(np.var(prediction.target_outputs, axis = 0)):
                    ax.axvline(x = t0 + prediction.valid_length(), linestyle = "--",
                               color = prediction_color, #"r",
                               linewidth = vert_linewidth,
                               label = "Valid Prediction Time")
            if ylabel is None:
                ax.set_ylabel(f"$x_{plot_dims[i]+1}$")
            else:
                ax.set_ylabel(ylabel)
            if prediction.reservoir_outputs.shape[0] > max_horizon:
                prev_left_lim = ax.get_xlim()[0]
                test_length = prediction.resync_inputs[lookback_length:, i].shape[0]
                new_left_lim = - test_length - abs(abs(prev_left_lim) - abs(test_length)) * max_horizon / prediction.reservoir_outputs.shape[0]
                ax.set_xlim(left = new_left_lim, right = max_horizon)
                
            if xlims is not None:
                ax.set_xlim(*xlims)
            if ylims is not None:
                ax.set_ylim(*ylims)
                
            if num_ticks is not None:
                ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
                ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
                #ax.set_xticks(np.linspace(min(initial_pts[:, 0]), max(initial_pts[:, 0]), num_ticks))
                #ax.set_yticks(np.linspace(min(initial_pts[:, 1]), max(initial_pts[:, 1]), num_ticks))
                
        if isinstance(legend_loc, tuple):
            axs[legend_ax].legend(loc = "center", bbox_to_anchor = legend_loc,
                                  ncols = n_legend_cols, frameon = frame_legend)
        else:
            axs[legend_ax].legend(loc = legend_loc, ncols = n_legend_cols, frameon = frame_legend)
        axs[-1].set_xlabel(xlabel) #"Time (Time Steps, $\\Delta t$)") #($\\tau_{Lyap}$)")
        #fig.suptitle("Valid Time: $T_{valid}=$" + f"{prediction.valid_length()}" + "$\\tau_{Lyap}$")
        
        if fig is not None:
            fig.patch.set_alpha(fig_alpha)
        
def plot_predict_result_errors(
        predictions:        Union[List[rc.PredictResult], rc.PredictResult],
        predicted_basins:   List = None,
        error_type:         str = "rmse",
        error_threshold:    float = None,
        normalize:          bool = True,
        max_horizon:        int = None,
        frame_legend:       bool = False,
        decimation:         int = None,
        legend_loc:         tuple = (.5, 1.1),
        legend_ax:          int = 0,
        n_legend_cols:      int = 4,
        font_size:          float = 15.,
        average:            str = None,
        spread:             str = None,
        incl_tvalid:        bool = True,
        fig:                mpl.figure.Figure = None,
        axes:               np.ndarray = None,
        bar_axes:           np.ndarray = None,
        xlabel:             str = "Time (Time Steps, $\\Delta t$)", #None,
        ylabel:             str = None,
        color:              Union[str, tuple] = "tab:blue",
        linewidth:          float = None,
        xlims:              tuple = None,
        ylims:              tuple = None,
        figsize:            tuple = None,
        num_ticks:          int = None,
        line_alpha:         float = 1,
        fig_alpha:          float = 1,
        vert_linewidth:     float = None,
        bar_alpha:          float = 1,
        bar_bins:           Union[list, np.ndarray, int] = None,
        t0:                 float = 0.,
        incorrect_color:    str = "black",
        colormap:           Union[dict, str, mpl.colors.LinearSegmentedColormap] = None,
        xticks:             Union[list, np.ndarray] = None,
        yticks:             Union[list, np.ndarray] = None,
        num_each:           bool = False,
        num_basins:         int = None
        ):
    
    if not isinstance(predictions, list):
        predictions = [predictions]
        
    #if decimation is not None:
    #    predictions = predictions[::decimation]
    if decimation is None:
        decimation = 1
    
    if error_type == "euclidean":
        errors = np.array([
            np.sqrt(np.sum(np.square(p.reservoir_outputs - p.target_outputs), axis = 1))
            for p in predictions
            ])
    elif error_type == "rmse":
        if normalize:
            errors = np.array([p.nrmse for p in predictions])
        else:
            errors = np.array([p.rmse for p in predictions])
            
    if predicted_basins is not None and colormap is not None:
        colors = [colormap(pb) for pb in predicted_basins]

    if max_horizon is None:
        max_horizon = errors.shape[0]        
        
    if bar_axes is not None:
        end_errs = np.array([e[-25:].mean() for e in errors])
        histogram_data = np.histogram(end_errs,
                                      bins = 1000,
                                      density = True)
        #print(np.array(histogram_data).shape)
        n, bins, patches = bar_axes.hist(
            end_errs,
            bins = bar_bins, #250, #"auto", #100,
            orientation = "horizontal",
            #density = True,
            weights = np.ones_like(end_errs) / len(end_errs),
            #density = False, #fill = False,
            #stacked = True,
            alpha = bar_alpha,
            color = color
            )
        print("Histogram Sum: ", np.sum(n))
        bar_axes.set_xlabel("Freq.")
    
    #if xlabel is None:
    #    xlabel = "Time (Time Steps, $\\Delta t$)"
    
    with mpl.rc_context({"font.size" : font_size}):
        if figsize is None:
            figsize = (12, 3)
        if axes is None:
            fig, axs = plt.subplots(1, 1,
                                    figsize = figsize, 
                                    sharex = True, constrained_layout = True)
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
        else:
            axs = axes
            if isinstance(axs, mpl.axes._axes.Axes):
                axs = [axs]
            for ax in axs[1:]:
                ax.sharex(axs[0])
            for ax in axs[:-1]:
                
                ax.tick_params(labelbottom = False)
        
        
        for i, ax in enumerate(axs):
            
            if average is None:
                already_plotted = np.zeros(num_basins)
                for ind in range(0, errors.shape[0], decimation):
                    if predicted_basins is not None and colormap is not None:
                        ind_color = colors[ind]
                    elif error_threshold is None:
                        ind_color = color
                    elif errors[ind][-25:].max() > error_threshold:
                        ind_color = incorrect_color #"tab:purple" #"tab:gray"
                    else:
                        ind_color = color
                    if num_each is not None:
                        """
                        if already_plotted[predicted_basins[ind]] < num_each:
                            ax.plot(t0 + np.arange(errors.shape[1]), errors[ind],
                                    c = ind_color, alpha = line_alpha, linewidth = linewidth)
                            already_plotted[predicted_basins[ind]] += 1
                        """
                        #print(ind)
                        #print(predicted_basins[ind])
                        if not np.isnan(predicted_basins[ind]):
                            if already_plotted[predicted_basins[ind]] == num_each:
                                ax.plot(t0 + np.arange(errors.shape[1]), errors[ind],
                                        c = ind_color, alpha = line_alpha, linewidth = linewidth)
                                already_plotted[predicted_basins[ind]] += 1
                            elif already_plotted[predicted_basins[ind]] < num_each:
                                already_plotted[predicted_basins[ind]] += 1
                        
                    else:
                        ax.plot(t0 + np.arange(errors.shape[1]), errors[ind],
                                c = ind_color, alpha = line_alpha, linewidth = linewidth)
                label = ""
            else:
                if average == "median":
                    avgd = np.median(errors, axis = 0)
                    label = "Median "
                elif average == "mean":
                    avgd = np.mean(errors, axis = 0)
                    label = "Mean "
                    
                if spread == "iqr":
                    upper_spread = np.quantile(errors, .75, axis = 0)
                    lower_spread = np.quantile(errors, .25, axis = 0)
                elif spread == "std":
                    std = np.std(errors, axis = 0)
                    upper_spread = avgd + std
                    lower_spread = avgd - std
        
                ax.plot(t0 + np.arange(errors.shape[1]), avgd, c = color, alpha = line_alpha,
                        linewidth = linewidth)
                if spread is not None:
                    ax.fill_between(t0 + np.arange(errors.shape[1]),
                                    y1 = lower_spread, y2 = upper_spread, color = color,
                                    alpha = .25)
            
            if ylabel is None:
                ax.set_ylabel("$\\varepsilon$")
            else:
                ax.set_ylabel(label + ylabel)
                    
            if xlims is not None:
                ax.set_xlim(*xlims)
            if ylims is not None:
                ax.set_ylim(*ylims)
                
            if num_ticks is not None:
                ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
                ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
                
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        
        if isinstance(legend_loc, tuple):
            axs[legend_ax].legend(loc = "center", bbox_to_anchor = legend_loc,
                                  ncols = n_legend_cols, frameon = frame_legend)
        else:
            axs[legend_ax].legend(loc = legend_loc, ncols = n_legend_cols, frameon = frame_legend)
        axs[-1].set_xlabel(xlabel)
        
        fig.patch.set_alpha(fig_alpha)
        
def plot_errors_histograms(
        predictions:        Union[List[rc.PredictResult], rc.PredictResult],
        predicted_basins:   List = None,
        error_type:         str = "rmse",
        error_threshold:    float = None,
        normalize:          bool = True,
        frame_legend:       bool = False,
        font_size:          float = 15.,
        fig:                mpl.figure.Figure = None,
        ax:                 np.ndarray = None,
        xlabel:             str = None,
        ylabel:             str = None,
        color:              Union[str, tuple] = "tab:blue",
        linewidth:          float = None,
        xlims:              tuple = None,
        ylims:              tuple = None,
        figsize:            tuple = None,
        num_ticks:          int = None,
        fig_alpha:          float = 1,
        vert_linewidth:     float = None,
        bar_alpha:          float = 1,
        bar_bins:           Union[list, np.ndarray, int] = None,
        xticks:             Union[list, np.ndarray] = None,
        yticks:             Union[list, np.ndarray] = None,
        orientation:        str = 'vertical',
        bar_error_scale:    str = "linear",
        bar_scale:          str = "linear",
        t0:                 int = -25,
        t1:                 int = None
        ):
    
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    if error_type == "euclidean":
        errors = np.array([
            np.sqrt(np.sum(np.square(p.reservoir_outputs - p.target_outputs), axis = 1))
            for p in predictions
            ])
    elif error_type == "rmse":
        if normalize:
            errors = np.array([p.nrmse for p in predictions])
        else:
            errors = np.array([p.rmse for p in predictions])

    if ax is not None:
        end_errs = np.array([e[t0: t1].max() for e in errors]) #mean() for e in errors])
        #print(np.array(histogram_data).shape)
        n, bins, patches = ax.hist(
            end_errs,
            bins = bar_bins, #250, #"auto", #100,
            orientation = orientation,
            #density = True,
            weights = np.ones_like(end_errs) / len(end_errs),
            alpha = bar_alpha,
            color = color
            )
        print("Histogram Sum: ", np.sum(n))
    if orientation == "vertical":
        ax.set_ylabel("Freq.")
        ax.set_xlabel("Maximum Error over Final $25$ Predicted Points, $\\varepsilon^{max}_{25}$")
        ax.set_xscale(bar_error_scale)
        ax.set_yscale(bar_scale)
        if error_threshold is not None:
            ax.axvline(error_threshold, c = "k", linestyle = "--")
        
        if xlims is not None:
            ax.set_ylim(*xlims)
        if ylims is not None:
            ax.set_xlim(*ylims)
        if num_ticks is not None:
            ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
            #ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
        if xticks is not None:
            ax.set_yticks(xticks)
        if yticks is not None:
            ax.set_xticks(yticks)
            
    elif orientation == "horizontal":
        ax.set_xlabel("Freq.")
        ax.set_ylabel("Error, $\\varepsilon$")
        ax.set_yscale(bar_error_scale)
        ax.set_xscale(bar_scale)
        if error_threshold is not None:
            ax.axhline(error_threshold, c = "k", linestyle = "--")
            
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)
        if num_ticks is not None:
            ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
            #ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
    
    fig.patch.set_alpha(fig_alpha)


#%% Training Data Construction

@dataclass
class MappingRC_TrainData:
	
	"""
	An object to store training data for the Mapping RC. Takes as arguments 
	a list of input signals and, optionally, any subset of some common targets.
	Properties allow construction of some common combined targets, and stores
	the number of samples.
	"""
	
	signals:			List[np.ndarray]
	parameters:		List[np.ndarray] = None
	weights:			List[np.ndarray] = None
	initial_states:	List[np.ndarray] = None
	final_states:	List[np.ndarray] = None
	
	def __post_init__(self):
		self.num_samples = len(self.signals)
	
	@property
	def weights_and_ri(self):
		if (self.initial_states, self.weights) != (None, None):
			return [np.concatenate((self.initial_states[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_ri not defined. Please ensure that both " \
				"initial_states and weights have been provided."
			logging.error(msg)
	
	@property
	def weights_and_rf(self):
		if (self.final_states, self.weights) != (None, None):
			return [np.concatenate((self.final_states[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_rf not defined. Please ensure that both " \
				"final_states and weights have been provided."
			logging.error(msg)
	
	@property
	def weights_and_params(self):
		if (self.parameters, self.weights) != (None, None):
			return [np.concatenate((self.parameters[j], self.weights[j]),
						  axis = 1) for j in range(self.num_samples)]
		else:
			msg = "weights_and_params not defined. Please ensure that both " \
				"parameters and weights have been provided."
			logging.error(msg)

class Library():
    
    '''
    An object to store a library of data for use with a LARC method.
    
    Args:
        data (list, np.ndarray) : A list of arrays, where each array is a 
            time-series (or sequential data structure) from a library member.
            If a single array is passed, it will be interpreted as a library of
            one data point. Dimension (num_members * num_time_steps * num_parameters).
        parameters (list) : A list in which each entry containings the 
            parameters of the library member in the corresponding entry of the
            data list. Dimension (num_members * num_parameters).
        parameter_labels (list): A list of num_parameters strings identifying
            the variables in the parameters list.
        data_generator (Callable) : A function to generate new library members.
            Should take as argument the parameters of an entry in the
            parameters list in the order they appear in that list. Should also
            take an argument "seed" to accept seed for random number generation.
        generator_args: A list of other arguments used by the data_generator
            routine, such as time-step, length, or transient_length.
        seeds (list): Seeds to be used with data_generator if data not provided.
    '''
    
    def __init__(
            self,
            data:               Union[List[np.ndarray], np.ndarray] = None,
            parameters:         List = None,
            parameter_labels:   Union[str, List[str]] = None,
            parameter_dynamics: Union[Callable, List[Callable]] = None,
            dynamics_args:      dict = {},
            data_generator:     Callable = None,
            generator_args:     dict = {},
            seed:               float = 1000,
            standardizer:       rc.Standardizer = None,
            standardize:        bool = False
            ):
        
        if isinstance(data, np.ndarray) and data is not None:
            self.data = [data]
        
        self.data = data
        self.parameters = parameters
        self.parameter_labels = parameter_labels
        if parameter_dynamics is not None and isinstance(parameter_dynamics, Callable):
            self.parameter_dynamics = [parameter_dynamics] * len(self.parameter_labels)
        else:
            self.parameter_dynamics = parameter_dynamics
        self.dynamics_args = dynamics_args
        self.generator_args = generator_args
        self.data_generator = data_generator
        self.standardizer = standardizer
        self.standardize = standardize
        if isinstance(parameters, list) and data is None:
            self.seeds = list(np.arange(seed, seed + len(parameters) + 1, 1))
        else:
            self.seeds = None
        
    def standardize_data(self):
        
        if self.standardizer is None:
            u = self.data[0]
            if len(self.data) > 1:
                u = functools.reduce(
                    lambda u, u_new: np.concatenate((u, u_new), axis = 0), self.data[1:])
            self.standardizer = rc.Standardizer(u = u, scale_factor = "max") #"var") #"max")
        
        self.data = [self.standardizer.standardize(u = signal) for signal in self.data]
        self.standardize = True
        
    def unstandardize_data(self):
        
        if self.standardizer is None:
            msg = "Cannot unstandardize data if standardizer has not been set, " \
                "and data has not already been standardized."
            logging.error(msg)
            raise(NotImplementedError(msg))
        
        self.data = [self.standardizer.unstandardize(u = signal) for signal in self.data]
    
    def generate_data(self):
        
        if self.parameter_dynamics is None:
            parameters = self.parameters
        else:
            parameters = [[
                dynamics(parameter = parameter, **self.dynamics_args)
                for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                for parameter_instance in self.parameters]
        
        self.data = [self.data_generator(
            **{self.parameter_labels[i] :
               param_i for i, param_i in enumerate(parameter_instance)},
            seed = seed, **self.generator_args)
            for parameter_instance, seed in zip(parameters, self.seeds)]
        
        if self.standardize:
            self.standardize_data()
            
            
    def add_datum(
            self,
            data:           np.ndarray = None,
            parameters:     list = None,
            seed:           Union[int, float] = None
            ):
        
        if isinstance(data, np.ndarray):
            if parameters is not None:
                if isinstance(self.parameters, list) and len(self.parameters) == len(self.data):
                    self.parameters = self.parameters + parameters
                    if isinstance(self.seeds, list):
                        self.seeds = self.seeds + [seed]
            self.data = self.data + [data]
                
        else:
            if parameters is not None:
                if isinstance(self.parameters, list) and len(self.parameters) == len(self.data):
                    self.parameters = self.parameters + parameters
                    if isinstance(self.seeds, list):
                        self.seeds = self.seeds + [seed]
                    if self.parameter_dynamics is not None:
                        parameters = [[
                            dynamics(parameter = parameter, **self.dynamics_args)
                            for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                            for parameter_instance in self.parameters]
                    self.data = self.data + [
                        self.data_generator(
                            **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameters)},
                            seed = seed, **self.generator_args)
                        ]
        
        if self.standardize:
            self.standardize_data()
                    
    def generate_grid(
            self,
            points:         Union[List, List[np.ndarray]] = None,
            ranges:         Union[List, List[tuple]] = None,
            seed:           float = 1000,
            scale:          str = "linear",
            incl_endpoint:  bool = True,
            dtype:          str = None,
            base:           float = 10.
            ):
        
        '''
        A routine to generate a grid in parameter space.
        
        Args:
            points (list): A list num_parameters long, where each entry contains all
                the sample values along that parameter axis.
            ranges (list): A list num_parameters long, containing in each entry a 
                3-tuple or 3-list of whose first entry is the lower bound of 
                sampled values along the corresponding axis, second entry is
                the upper bound, and third is the number of samples to return.
            seed (int): A seed to generate data samples.
            scale (str): Defaults to linear. "log" for equal spacing on a 
                logarithm scale.
            incl_endpoint (bool): Defaults to true. If true, returns the end
                point of each range. If false, the endpoint is discarded.
            dtype (str): Selected the data type of returned entries in the 
                parameter list.
            base (float): Only used if scale = "log". The base of logarithms
                taken.
        '''
        
        if points is not None:
            if ranges is not None:
                msg = "points and ranges both provided. Ignoring ranges."
                logging.warning(msg)
        
        elif ranges is not None:
            if scale == "linear":
                points = [np.linspace(rng[0], rng[1], rng[2], endpoint = incl_endpoint)
                          for rng in ranges]
            elif scale == "log":
                points = [np.logspace(rng[0], rng[1], rng[2], endpoint = incl_endpoint,
                                      base = base) for rng in ranges]
            else:
                msg = "Tyring to use ranges to return parameter grid, but no \
                    appropriate scale provided. Please pass 'linear' or 'log'."
                logging.warning(msg)
        
        else:
            msg = "Please provided either points or ranges to construct points."
            logging.error(msg)
            
        parameters = list(itertools.product(*points))
        seeds = list(np.arange(seed, seed + len(parameters) + 1, 1))
        
        if self.parameters is None and self.data is None:
            self.parameters = parameters
            self.seeds = seeds
            self.generate_data()
            
        elif (self.parameters is not None and self.data is not None) and \
            len(self.parameters) == len(self.data):
            self.parameters = self.parameters + parameters
            self.data = self.data + [
                self.data_generator(
                    **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameter_instance)},
                    seed = seed, **self.generator_args)
                for parameter_instance, seed in zip(parameters, seeds)
                ]
            if self.standardize:
                self.standardize_data()
            
        elif self.parameters is not None and self.data is None:
            self.parameters = self.parameters + parameters
            self.seeds = self.seeds + seeds
            self.generate_data()
            
        else: #if self.parameters is None and self.data is not None
            self.data = self.data + [
                self.data_generator(
                    **{self.parameter_labels[i] : param_i for i, param_i in enumerate(parameter_instance)},
                    seed = seed, **self.generator_args)
                for parameter_instance, seed in zip(parameters, seeds)
                ]
            if self.standardize:
                self.standardize_data()
            
    def set_library_RCs(
            self,
            pred_esn:           rc.ESN,
            transient_length:   Union[int, List[int]],
            train_args:         dict = {}
            ):
        
        """
        Given a list of input time-series, return a list of corresponding output 
        layer weights trained for prediction with the provided ESN object.
        """
        
        if self.data is None:
            self.generate_data()
        
        if isinstance(transient_length, int):
            transient_length = [transient_length] * len(self.data)
            
        weights = [pred_esn.train(transient_length = transient_length[i],
                                  inputs = self.data[i], **train_args
                                  ).weights for i in range(len(self.data))]
        self.weights = weights
        self.esn = pred_esn
        
        return weights
    
    def plot_parameter_dynamics(self, num_samples: int = None):
        
        if num_samples is None:
            num_samples = len(self.parameters)
        else:
            num_samples = min(num_samples, len(self.parameters))
            
        if self.parameter_dynamics is not None:
            parameters = [[
                dynamics(parameter = parameter, **self.dynamics_args)
                for parameter, dynamics in zip(parameter_instance, self.parameter_dynamics)]
                for parameter_instance in self.parameters]
            
        for sample in range(num_samples):
            parameter_case = parameters[sample]
            fig, ax = plt.subplots(len(parameter_case), 1, sharex = True,
                                   constrained_layout = True)
            if len(parameter_case) == 1:
                ax = [ax]
            for ind, param in enumerate(parameter_case):
                ax[ind].plot(param)
                ax[ind].set_ylabel(self.parameter_labels[ind])
            ax[-1].set_xlabel("Time Steps")
            plt.show()
            
    def plot_data(
            self,
            num_samples:    int = None,
            color:          Union[str, tuple] = None,
            figsize:        tuple = None,
            time_range:     Union[list, tuple] = (None, None)
            ):
        
        if num_samples is None:
            num_samples = len(self.data)
        else:
            num_samples = min(num_samples, len(self.data))
            
        for sample in range(num_samples):
            data_sample = self.data[sample]
            fig, ax = plt.subplots(
                data_sample.shape[1], 1, sharex = True,
                figsize = figsize, constrained_layout = True
                )
            if data_sample.shape[1] == 1:
                ax = [ax]
            for ind in range(data_sample.shape[1]):
                ax[ind].plot(
                    data_sample[time_range[0]: time_range[1], ind],
                    color = color
                    )
                ax[ind].set_ylabel(f"$x_{ind}$")
            ax[-1].set_xlabel("Time Steps")
            plt.show()
    
    def copy(self):
        
        """
        A routine to return a copy of a provided Library object.
        """
        
        new_library = Library(
            data = self.data,
            parameters = self.parameters,
            parameter_labels = self.parameter_labels,
            data_generator = self.data_generator,
            generator_args = self.generator_args
            )
        new_library.seeds = self.seeds
        '''
        if hasattr(self, "weights"):
            new_library.weights = self.weights
        if hasattr(self, "esn"):
            new_library.esn = self.esn
        '''
        for attr in list(self.__dict__.keys()):
            if not hasattr(new_library, attr):
                setattr(new_library, attr, getattr(self, attr))
            
        return new_library
    
    def save(self, save_loc, safe_save = False, file_name = None, reduce = False):
		
        """        
		Saves the library in a provided directory.
		
		Args:
			save_loc (str): The absolute or relative path to the folder.
            file_name (str): The name of the file in which the Run_Result will
                             be stored. If None, defaults to "run_data.pickle".
			safe_save (bool): If False, will overwrite existing files and
                              folders.
                              Otherwise, will raise an exception if saving
                              would overwrite anything.
            reduce (bool): If True, set self.data, self.esn, and self.weights
                           to None to save space.
		"""
        
        save_copy = self.copy() #copy_library(self)
        save_copy.data_generator = None
        
        if reduce:
            save_copy.data = None
            if hasattr(save_copy, "esn"):
                save_copy.esn = None
            if hasattr(save_copy, "weights"):
                save_copy.weights = None
                
        if file_name is None:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            if os.path.isdir(save_loc):
                if safe_save:
                    msg = f"Already folder or file at '{save_loc}' and " \
                        "safe_save is True."
                    logging.error(msg)
                    raise FileExistsError(msg)
                else:
                    shutil.rmtree(save_loc)
                    msg = f"Already a folder or file at '{save_loc}' but " \
                        "safe_save is False; deleting the existing " \
                        "files and folders."
                    logging.info(msg)
                    
            os.makedirs(save_loc)
            with open(os.path.join(save_loc, "library.pickle"), 'wb') as temp_file:
                pickle.dump(save_copy, temp_file)
                
        else:
            # Check if the path exists
            # Overwrite it if safe_save is False; raise Exception if True.
            head, tail = os.path.split(os.path.join(save_loc, file_name))
            if os.path.isdir(head):
                if os.path.exists(os.path.join(save_loc, file_name + ".pickle")):
                    if safe_save:
                        msg = f"Already folder or file at '{save_loc}' and " \
                            "safe_save is True."
                        logging.error(msg)
                        raise FileExistsError(msg)
                    else:
                        msg = f"Already a folder or file at '{save_loc}' but " \
                              "safe_save is False; deleting the existing " \
                              "files and folders."
                        logging.info(msg)
                        
            else:
                os.makedirs(head)
            
            with open(os.path.join(save_loc, file_name + ".pickle"), 'wb') as temp_file:
                pickle.dump(save_copy, temp_file)

'''
def copy_library(
        library : Library
        ):
    
    """
    A routine to return a copy of a provided Library object.
    """
    
    new_library = Library(
        data = library.data,
        parameters = library.parameters,
        parameter_labels = library.parameter_labels,
        data_generator = library.data_generator,
        generator_args = library.generator_args
        )
    new_library.seeds = library.seeds
    if hasattr(library, "weights"):
        new_library.weights = library.weights
    if hasattr(library, "esn"):
        new_library.esn = library.esn
        
    return new_library
'''
    
def get_open_loop_targets(
		target_series:		np.ndarray,
		horizon:			int
		):
	
	"""
	Creates training targets for open-loop prediction of predict_length steps.
	"""
	
	len_series = target_series.shape[0]
	target_series = target_series.reshape((len_series, -1))
	
	open_loop_targets = np.zeros((target_series.shape[0] - horizon,
							   target_series.shape[1]))
	open_loop_targets = target_series[1: - horizon + 1]
	for step in range(2, horizon):
		open_loop_targets = np.hstack((
			open_loop_targets,
			target_series[step: - horizon + step].reshape((-1, 1))
			))
	open_loop_targets = np.hstack((open_loop_targets,
								target_series[horizon:].reshape((-1, 1))))
		
	return open_loop_targets

def convert_open_loop_predict_result(
		predict_result:		rc.PredictResult
		):
	
	
	
	return predict_result

def get_library_weights(
        pred_esn:           rc.ESN,
        inputs:             Union[np.ndarray, List[np.ndarray]],
        transient_length:   Union[int, List[int]],
        train_args:         dict = {}
        ):
    
    """
    Given a list of input time-series, return a list of corresponding output 
    layer weights trained for prediction with the provided ESN object.
    """
    
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    if isinstance(transient_length, int) or len(transient_length.shape) == 0:
        transient_length = [transient_length] * len(inputs)

    return [pred_esn.train(transient_length = transient_length[i],
                           inputs = inputs[i], **train_args
                           ).weights for i in range(len(inputs))]
	
def extract_MRS_training_data(
        library_signals:        List[np.ndarray],
        tshort:                 int,
        esn:                    Union[rc.ESN, None],
        transient:              Union[int, List[int], None],
        library_targets:        Union[list[np.ndarray], np.ndarray, None] = None,
        parameters:             Union[List[int], List[float], List[np.ndarray]] = None,
        regression:             Callable = regressions.tikhonov(),
        feature_function:       Union[features.ESNFeatureBase, Callable] = features.StatesOnly(),
        batch_size:             int = None,
        batch_length:           int = None,
        start_time_separation:  int = 1,
        incl_weights:           bool = True,
        incl_initial_states:    bool = False,
        incl_final_states:      bool = False,
        open_loop_horizon:      int = None,
        future_refit_len:       int = None,
        refit_regression:       Callable = lambda prior: regressions.batched_ridge(prior_guess = prior)
		):
	
    """
    Extracts MRS-method training data (short signals and (r_0, W_out) pairs) from
    a list of provided library signals.
    """
	
    if parameters is not None:
        for index, entry in enumerate(parameters):
            if type(entry) in [float, int, np.int32, np.int64,
                               np.float32, np.float64]:
                parameters[index] = np.array([[entry]])
				
    if library_targets is not None:
        if not isinstance(library_targets, list):
            library_targets = [library_targets]

    if isinstance(transient, int) or len(transient.shape) == 0:
        transient = [transient] * len(library_signals)
    elif isinstance(transient, list) and len(transient) != len(library_signals):
        msg = "transient must have the same length as library_signals."
        logging.error(msg)
	
    short_sigs = list()
    sub_ws = list()
    sub_r0s = list()
    sub_r0s_resync = list()
    sub_params = list()
    regression_args = inspect.signature(regression).parameters
    for i in range(len(library_signals)):
        shorts_i = windower(library_signals[i][transient[i]:], tshort, axis = 0)
        shorts_i = [short.T for short in shorts_i]
        if isinstance(future_refit_len, int) and future_refit_len > 0:
            shorts_i = shorts_i[:-future_refit_len]
            shorts_inds = windower(np.arange(library_signals[i].shape[0])[transient[i]:],
                                   tshort, axis = 0)[:-future_refit_len]
            shorts_inds = [inds[-1] for inds in shorts_inds]
            refits_i = [library_signals[i][:j + future_refit_len] for j in shorts_inds]
            refits_i = [refit.reshape((-1, library_signals[i].shape[1])) for refit in refits_i]
        num_shorts_i = len(list(shorts_i))
		
        if esn is not None and transient is not None:
            train_args = {
    			"transient_length" : transient[i],
    			"feature_function" : feature_function,
    			"regression" : regression
    			}
            if open_loop_horizon is None:
                train_args["inputs"] = library_signals[i]
            else:
                train_args["inputs"] = library_signals[i][:-open_loop_horizon]
            if library_targets is not None:
                train_args["target_outputs"] = library_targets[i]
            if "VS_T" in regression_args and "SS_T" in regression_args:
                train_args["batch_size"] = batch_size
                train_args["batch_length"] = batch_length
    			
            train_result = esn.train(**train_args)
            w_out = train_result.weights
		
        for j in range(num_shorts_i):
            if(j % start_time_separation == 0):
                if esn is not None:
                    short_sigs.append(shorts_i[j])
                if esn is not None and incl_weights:
                    if future_refit_len is None:
                        sub_ws.append(w_out.reshape(1, -1))
                    elif isinstance(future_refit_len, int) and future_refit_len > 0:
                        train_args = {
                			"transient_length" : shorts_inds[j],
                			"feature_function" : feature_function,
                			"regression" : refit_regression(prior = w_out),
                        "batch_size" : batch_size,
                        "batch_length" : batch_length
                			}
                        if open_loop_horizon is None:
                            train_args["inputs"] = refits_i[j]
                        else:
                            train_args["inputs"] = refits_i[j][:-open_loop_horizon]
                        if library_targets is not None:
                            train_args["target_outputs"] = library_targets[i][
                                :shorts_inds[j] + future_refit_len]
                        sub_ws.append(esn.train(**train_args).weights.reshape(1, -1))
                                      #- w_out.reshape((1, -1)))
                if parameters is not None:
                    sub_params.append(parameters[i])
		
        if esn is not None and incl_final_states:
            r0s = train_result.states[transient[i] - 1 + tshort - 1:]
            for j in range(r0s.shape[0]):
                if(j % start_time_separation == 0):
                    sub_r0s.append(r0s[j].reshape(1, -1))
		
        if esn is not None and incl_initial_states:
            if tshort > 1:
                r0s_resync = train_result.states[transient[i] - 1: - tshort + 1]
            else:
                r0s_resync = train_result.states[transient[i] - 1:]
            for j in range(r0s_resync.shape[0]):
                if(j % start_time_separation == 0):
                    sub_r0s_resync.append(r0s_resync[j].reshape(1, -1))
	
    return MappingRC_TrainData(signals = short_sigs, parameters = sub_params,
							weights = sub_ws, initial_states = sub_r0s_resync,
							final_states = sub_r0s)

