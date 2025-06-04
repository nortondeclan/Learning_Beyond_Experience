#%% Import Statements

import rescompy as rc
import rescompy.features as features
import matplotlib.pyplot as plt
import numpy as np
import rc_helpers as rch
import matplotlib as mpl
import os
import logging
import shutil
import pickle
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union, List
from numpy.random import default_rng

windower = np.lib.stride_tricks.sliding_window_view

#%% Class Definitions
@dataclass
class basin_prediction:
    
    predictor:          str
    initial_pts:        Union[int, list, np.ndarray]
    predicted_finals:   Union[int, list, np.ndarray]
    true_finals:        Union[int, list, np.ndarray]
    prediction:         rc.PredictResult = None
    sm_prediction:      rc.PredictResult = None
    performance_metric: Union[int, list, np.ndarray] = None
    metric_name:        str = None
    
#%% Base Predictor Class
class basin_predictor_base(ABC):
 
    @abstractmethod
    def train(
            self,
            library:                rch.Library,
            test_length:            int,
            train_args:             dict = {},
            safe_train:             bool = False,
            return_result:          bool = False,
            reduce:                 bool = False
            ):
        pass
    
    @abstractmethod
    def predict(
            self,
            library:                rch.Library,      
            test_length:            int,
            return_predictions:     bool = True,
            save_predictions:       bool = False,
            save_loc:               str = None,
            safe_save:              bool = False,
            file_name:              str = None,
            reduce:                 bool = False
            ):
        pass
    
    @abstractmethod
    def copy(self):
        pass
    
    @abstractmethod
    def save(
            self,
            save_loc:               str,
            safe_save:              bool = False,
            file_name:              str = None,
            reduce_fully:           bool = False,
            reduce_states:          bool = False
            ):
        pass

#%% Batch Basin Predictor

class batch_predictor(basin_predictor_base):
    
    def __init__(
            self,
            reservoir:          rc.ESN,
            train_result:       rc.TrainResult = None,
            feature_function:   Union[Callable, features.ESNFeatureBase] = None,
            ):
        
        '''

        Parameters
        ----------
        reservoir : rc.ESN
            DESCRIPTION.
        train_result : rc.TrainResult, optional
            DESCRIPTION. The default is None.
        feature_function : Union[Callable, features.ESNFeatureBase], optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.reservoir = reservoir
        self.train_result = train_result
        if hasattr(train_result, "feature_function"):
            self.feature_function = train_result.feature_function
        else:
            self.feature_function = None
        
    def train(
            self,
            library:            rch.Library,
            train_args:         dict = {},
            safe_train:         bool = False,
            return_result:      bool = False,
            reduce:             bool = False,
            noise_amp:          float = None,
            noise_seed:         int = 9999
            ):
        
        '''

        Parameters
        ----------
        library : rch.Library
            DESCRIPTION.
        train_args : dict, optional
            DESCRIPTION. The default is {}.
        safe_train : bool, optional
            DESCRIPTION. The default is False.
        return_result : bool, optional
            DESCRIPTION. The default is False.
        reduce : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
            
        if self.train_result is not None and safe_train:
            print("safe_train is True. Cannot override existing train_result" +
                  " Retaining existing train_result")
            
        else:
            
            if self.train_result is not None and not safe_train:
                print("Overriding existing train_result.")
            
            if noise_amp is not None:
                rng = np.random.default_rng(noise_seed)
                u = library.data[0]
                if len(library.data) > 1:
                    u = functools.reduce(
                        lambda u, u_new: np.concatenate((u, u_new), axis = 0),
                        library.data[1:])
                sig_amp = np.sqrt(np.mean(np.square(u), axis = 0))
                train_inputs = [signal + rng.normal(0, noise_amp * sig_amp, signal.shape)
                                for signal in library.data]
            else:
                train_inputs = library.data
            
            self.train_result = self.reservoir.train(
                inputs = [signal[:-1] for signal in train_inputs],
                target_outputs = [signal[1:] for signal in library.data],
                **train_args
                )
            self.feature_function = self.train_result.feature_function
        
        if reduce:
            self.train_result = rch.reduce_train_result(self.train_result)
        
        if return_result:
            return self.train_result
        
    def predict(
            self,
            library:                rch.Library,      
            test_length:            int,
            return_predictions:     bool = True,
            save_predictions:       bool = False,
            save_loc:               str = None,
            safe_save:              bool = False,
            file_name:              str = None,
            reduce_fully:           bool = False,
            reduce_states:          bool = False,
            r0_perturbation:        float = 0,
            open_loop:              bool = False,
            mapper_func:            Callable = rc.default_mapper,
            metric_func:            Callable = None,
            end_processing:         Callable = None,
            ):
        
        '''

        Parameters
        ----------
        library : rch.Library
            DESCRIPTION.
        test_length : int
            DESCRIPTION.
        return_predictions : bool, optional
            DESCRIPTION. The default is True.
        save_predictions : bool, optional
            DESCRIPTION. The default is False.
        save_loc : str
            DESCRIPTION.
        safe_save : bool, optional
            DESCRIPTION. The default is False.
        file_name : str, optional
            DESCRIPTION. The default is None.
        reduce : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        predict_args = dict(
            train_result = self.train_result,
            initial_state = np.zeros(self.reservoir.size),
            mapper = mapper_func
            )
        if not hasattr(self.train_result, "feature_function"):
            predict_args["feature_function"] = self.feature_function
        
        predictions = []
        for signal, finals in zip(library.data, library.final_parameters):
            
            if open_loop:
                prediction = self.reservoir.predict(
                    resync_signal = signal[:test_length],
                    inputs = signal[test_length:],
                    target_outputs = signal[test_length:],
                    **predict_args
                    )            
            elif r0_perturbation == 0:
                prediction = self.reservoir.predict(
                    resync_signal = signal[:test_length],
                    target_outputs = signal[test_length:],
                    **predict_args
                    )
            else:
                resync_state = self.reservoir.get_states(
                    inputs = signal[:test_length],
                    initial_state = np.zeros(self.reservoir.size)
                    )[-1]
                predict_args["initial_state"] = resync_state + np.random.normal(0, r0_perturbation, self.reservoir.size)
                prediction = self.reservoir.predict(
                    target_outputs = signal[test_length:],
                    **predict_args
                    )
            
            if library.standardize:
                #true_finals = finals/library.standardizer.scale[:2]
                #true_finals += -library.standardizer.shift[:2]
                if end_processing is None:
                    pred_finals = library.standardizer.unstandardize(
                        u = prediction.reservoir_outputs)[-1].flatten()[:2]
                else:
                    pred_finals = end_processing(library.standardizer.unstandardize(
                        u = prediction.reservoir_outputs)).flatten()[:2]
                result_args = dict(
                    initial_pts = library.standardizer.unstandardize(
                        u = signal)[0].flatten()[:2],
                    predicted_finals = pred_finals,
                    #predicted_finals = library.standardizer.unstandardize(
                    #    u = prediction.reservoir_outputs)[-1].flatten()[:2],
                    true_finals = finals #true_finals
                )
            else:
                if end_processing is None:
                    pred_finals = prediction.reservoir_outputs[-1].flatten()[:2]
                else:
                    pred_finals = end_processing(prediction.reservoir_outputs).flatten()[:2]
                result_args = dict(
                    initial_pts = signal[0].flatten()[:2],
                    predicted_finals = pred_finals,
                    #predicted_finals = prediction.reservoir_outputs[-1].flatten()[:2],
                    true_finals = finals
                    )
            if not reduce_fully:
                if library.standardize:
                    prediction.reservoir_outputs = library.standardizer.unstandardize(
                        u = prediction.reservoir_outputs)
                    prediction.target_outputs = library.standardizer.unstandardize(
                        u = prediction.target_outputs)
                    prediction.resync_inputs = library.standardizer.unstandardize(
                        u = prediction.resync_inputs)
                    prediction.resync_outputs = library.standardizer.unstandardize(
                        u = prediction.resync_outputs)
                result_args["prediction"] = prediction
            if metric_func is not None:
                if hasattr(metric_func, "name"):
                    result_args["metric_name"] = metric_func.name
                result_args["performance_metric"] = metric_func(prediction)
                
            pred_i = basin_prediction(predictor = "batch", **result_args)
            if reduce_states and not reduce_fully:
                pred_i.prediction = rch.reduce_prediction(pred_i.prediction)
            
            predictions.append(pred_i) #basin_prediction(predictor = "batch", **result_args))
        
        """
        if reduce_states and not reduce_fully:
            for prediction in predictions:
                prediction.prediction = rch.reduce_prediction(prediction.prediction)
        """
        
        if save_predictions:
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
                with open(os.path.join(save_loc, "_predictions.pickle"), 'wb') as temp_file:
                    pickle.dump(predictions, temp_file)
                    
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
                    pickle.dump(predictions, temp_file)

        if return_predictions:
            return predictions            
            
    def copy(self):
        
        return batch_predictor(
            reservoir = self.reservoir,
            train_result = self.train_result,
            feature_function = self.feature_function
            )
    
    def save(
            self,
            save_loc:       str,
            safe_save:      bool = False,
            file_name:      str = None,
            reduce:         bool = False
            ):
		
        """        
		Saves the sm_classifier in a provided directory.
		
		Args:
			save_loc (str): The absolute or relative path to the folder.
            file_name (str): The name of the file in which the sm_classifier will
                             be stored. If None, defaults to "run_data.pickle".
			safe_save (bool): If False, will overwrite existing files and
                              folders.
                              Otherwise, will raise an exception if saving
                              would overwrite anything.
            reduce (bool): If True, set self.data, self.esn, and self.weights
                           to None to save space.
		"""
        
        save_copy = self.copy()
        
        if reduce:
            save_copy.train_result = rch.reduce_train_result(save_copy.train_result)
                
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

#%% Construct a Library

def get_fixed_basin_ICs(
        num_ICs:                int,
        trajectory_generator:   Callable,
        get_basin:              Callable,
        seed:                   int,
        fp_id:                  int = 0,
        IC_rng:                 tuple = (-1, 1),
        end_processing:         Callable = None
        ):
    
    if end_processing is None:
        end_processing = lambda x: x[-1][:2] #trajectory[-1][:2]
    
    usable_ICs = []
    num_parameters = len(trajectory_generator.parameter_labels)
    ic_rng = default_rng(seed)
    while len(usable_ICs) < num_ICs:
        IC = ic_rng.uniform(IC_rng[0], IC_rng[1], num_parameters)
        args = {trajectory_generator.parameter_labels[i] : IC[i]
                for i in range(num_parameters)}
        trajectory = trajectory_generator(**args, seed = seed)
        fp = get_basin(end_processing(trajectory)) #trajectory[-1][:2])
        if fp == fp_id:
            usable_ICs.append(IC)
            
    return usable_ICs

def get_fixed_basin_ICs_avgd(
        num_ICs:                int,
        trajectory_generator:   Callable,
        get_basin:              Callable,
        seed:                   int,
        fp_id:                  int = 0,
        IC_rng:                 tuple = (-1, 1)
        ):
    
    usable_ICs = []
    num_parameters = len(trajectory_generator.parameter_labels)
    ic_rng = default_rng(seed)
    while len(usable_ICs) < num_ICs:
        IC = ic_rng.uniform(IC_rng[0], IC_rng[1], num_parameters)
        args = {trajectory_generator.parameter_labels[i] : IC[i]
                for i in range(num_parameters)}
        trajectory = trajectory_generator(**args, seed = seed)
        fp = get_basin(trajectory[:, 2].mean())
        if fp == fp_id:
            usable_ICs.append(IC)
            
    return usable_ICs

def get_lib(
        get_generator:          Callable,
        get_basin:              Callable = None,
        generator_args:         dict = {},
        lib_size:               int = None,
        lib_length:             int = None,
        basin_check_length:     int = None,
        seed:                   int = None,
        train_basin:            float = None,
        grid:                   bool = False,
        file_name:              str = None,
        forward_length:         int = None,
        backward_length:        int = None,
        flip_backwards_traj:    bool = True,
        IC_rng:                 tuple = (-1, 1),
        standardizer:           rc.Standardizer = None,
        standardize:            bool = False,
        end_processing:         Callable = None
        ):
    
    if end_processing is None:
        end_processing = lambda x: x[-1, :2]
    
    if lib_length is not None and (forward_length, backward_length) == (None, None):
        forward_length = lib_length
    
    if file_name is None:
        
        forward_generator = get_generator(
            return_length = forward_length,
            **generator_args
            )
        if train_basin is not None:
            check_generator = get_generator(
                return_length = basin_check_length,
                **generator_args
                )
        backward_generator = get_generator(
            return_length = backward_length,
            direction = 'backward',
            **generator_args
            )
    
    if file_name is not None:
        with open(file_name, "rb") as tmp_file:
            library = pickle.load(tmp_file)
            library.final_parameters = [np.array(list(end_processing(series))) #series[-1, :2]))
                                        for series in library.data]
    elif grid:
        x_rng = (IC_rng[0], IC_rng[1], int(np.sqrt(lib_size)))
        y_rng = (IC_rng[0], IC_rng[1], int(np.sqrt(lib_size)))
        if forward_length is not None:
            library = rch.Library(
                data = None,
                parameters = None,
                parameter_labels = forward_generator.parameter_labels, #["x0", "y0"],
                data_generator = forward_generator, #trajectory_generator,
                generator_args = {}, #"transient_length" : transient_length,
                                  #"return_length" : lib_length + transient_length,
                                  #"return_dims" : return_dims},
                seed = seed,
                standardize = standardize,
                standardizer = standardizer
                )
            library.generate_grid(
                ranges = [x_rng, y_rng],
                seed = seed
                )
            library.final_parameters = [np.array(list(end_processing(series))) #series[-1, :2]))
                                        for series in library.data]
        if backward_length is not None:
            if forward_length is not None:
                b_standardizer = library.standardizer
            else:
                b_standardizer = standardizer
            back_library = rch.Library(
                data = None,
                parameters = None,
                parameter_labels = backward_generator.parameter_labels, #["x0", "y0"],
                data_generator = backward_generator, #trajectory_generator,
                generator_args = {}, #"transient_length" : transient_length,
                                  #"return_length" : lib_length + transient_length,
                                  #"return_dims" : return_dims},
                seed = seed,
                standardize = standardize,
                standardizer = b_standardizer # May need enforce consistent with forward standardization
                )
            back_library.generate_grid(
                ranges = [x_rng, y_rng],
                seed = seed
                )
            back_library.final_parameters = [np.array(list(end_processing(series))) #series[-1, :2]))
                                             for series in back_library.data]
    else:
        if train_basin is None:
            train_ICs = list(default_rng(seed).uniform(IC_rng[0], IC_rng[1], (lib_size, 2)))
        else:
            train_ICs = get_fixed_basin_ICs(
                num_ICs = lib_size,
                trajectory_generator = check_generator,
                seed = seed,
                fp_id = train_basin,
                get_basin = get_basin,
                IC_rng = IC_rng,
                end_processing = end_processing
                )
        if forward_length is not None:
            library = rch.Library(
                data = None,
                parameters = train_ICs,
                parameter_labels = forward_generator.parameter_labels, #["x0", "y0"],
                data_generator = forward_generator, #trajectory_generator,
                generator_args = {}, #"transient_length" : transient_length,
                                  #"return_length" : lib_length + transient_length,
                                  #"return_dims" : return_dims},
                seed = seed,
                standardize = standardize,
                standardizer = standardizer
                )
            library.generate_data()
            library.final_parameters = [np.array(list(end_processing(series))) #series[-1, :2]))
                                        for series in library.data]
        
        if backward_length is not None:
            back_library = rch.Library(
                data = None,
                parameters = train_ICs,
                parameter_labels = backward_generator.parameter_labels, #["x0", "y0"],
                data_generator = backward_generator, #trajectory_generator,
                generator_args = {}, #"transient_length" : transient_length,
                                  #"return_length" : lib_length + transient_length,
                                  #"return_dims" : return_dims},
                seed = seed,
                standardize = standardize,
                standardizer = standardizer
                )
            back_library.generate_data()
            back_library.final_parameters = [np.array(list(end_processing(series))) #series[-1, :2]))
                                             for series in back_library.data]
    
    if forward_length is None:
        if flip_backwards_traj:
            back_library.data = [datum[::-1] for datum in back_library.data]
        library == back_library.copy()
    elif backward_length is not None:
        back_library.data = [datum[::-1] for datum in back_library.data]
        library.data = [np.concatenate((b_datum[:-1], f_datum))
                        for b_datum, f_datum in zip(back_library.data, library.data)]
    
    if get_basin is not None:
        library.get_basin = get_basin
        
    if library.standardize:
        final_parameters = np.array(library.final_parameters) / library.standardizer.scale[:2]
        final_parameters -= library.standardizer.shift[:2]
        library.final_parameters = list(final_parameters)
    
    return library

#%% Check Fixed Points
def get_attractor(
        fixed_points:       np.ndarray, # = None,
        energy_func:        Callable = None,
        energy_args:        dict = {},
        distance_threshold: float = None, #.05
        return_coords:      bool = False,
        energy_barrier_loc: Union[List, np.ndarray] = np.zeros(4),
        use_energies:       bool = True,
        visible_dimensions: Union[int, List[int]] = None,
        check_if_fixed_pt:  bool = True
        ):
    
    if visible_dimensions is None:
        visible_dimensions = [i for i in range(fixed_points[0].shape[0])]
    
    def nearest(
            final_loc:  np.ndarray
            ):
        
        num_fps = len(fixed_points)
        dists = np.array([
            np.sqrt(np.sum(np.square(
                final_loc - fixed_points[i, np.array(visible_dimensions)[np.array(visible_dimensions) < fixed_points.shape[1]]])))
            for i in range(num_fps)
            ])
        #for i in range(num_fps):
        #    print(fixed_points[i, visible_dimensions])
        #print(dists)
        #increment = 1./(num_fps - 1.)
        
        if use_energies:
            energy = energy_func(state = final_loc, **energy_args)
            barrier_energy = energy_func(energy_barrier_loc, **energy_args)
            if energy >= barrier_energy:
                return np.nan
            elif return_coords:
                return fixed_points[np.argmin(dists)]
            else:
                return np.argmin(dists)  
        
        elif distance_threshold is not None:
            if dists[np.argmin(dists)] > distance_threshold:
                return np.nan
            elif return_coords:
                return fixed_points[np.argmin(dists)]
            else:
                return np.argmin(dists)
        
        else:
            if return_coords:
                return fixed_points[np.argmin(dists)]
            else:
                return np.argmin(dists)
    
    return nearest

def basin_volume_error(
        basin_predictions:  List = None,
        true_basins:        Union[List, np.ndarray] = None,
        pred_basins:        Union[List, np.ndarray] = None,
        ignore_true_nans:   bool = True,
        get_basin:          Callable = None
        ):
    
    if basin_predictions is not None and get_basin is not None:
        true_basins = [get_basin(p.true_finals) for p in basin_predictions]
        pred_basins = [get_basin(p.predicted_finals) for p in basin_predictions]
    
    if isinstance(true_basins, list):
        true_basins = np.array(true_basins) #, dtype = int)
        
    if isinstance(pred_basins, list):
        pred_basins = np.array(pred_basins) #, dtype = int)
        
    if ignore_true_nans:
        true_basins = true_basins[np.argwhere(~np.isnan(true_basins))].flatten()
        pred_basins = pred_basins[np.argwhere(~np.isnan(true_basins))].flatten()
     
    """
    print(true_basins.shape)
    print(pred_basins.shape)
    
    true_basins = true_basins.astype(int)
    for i in range(len(pred_basins)):
        if not np.isnan(pred_basins[i]):
            pred_basins[i] = int(pred_basins[i])
    """
        
    true_basin_ids, true_counts = np.unique(true_basins, return_counts = True)
    pred_counts = np.array([len(np.where(pred_basins == basin_id)[0]) for basin_id in true_basin_ids])
    
    pred_errors = np.abs(true_counts - pred_counts)
    
    error = np.sum(pred_errors)
    if len(np.where(np.isnan(pred_basins))[0]) > 1:
        num_unphysical = len(np.where(np.isnan(pred_basins))[0])
    else:
        num_unphysical = 0
    
    """
    print(true_counts)
    print(pred_counts)
    print(len(true_basins))
    print(error)
    print(num_unphysical)
    """    
    
    error = (error + num_unphysical) / (2 * len(true_basins))
    fraction_unphysical = num_unphysical / len(true_basins)
    all_basins_predicted = min(true_counts) / len(true_basins)
    pred_errors = pred_errors/len(true_basins)
    
    return error, pred_errors, fraction_unphysical, all_basins_predicted
    

#%% Plotting

def basin_heatmap(
        fixed_points:   np.ndarray,
        library:        rch.Library = None,
        get_fixed_pt:   Callable = None,
        initial_pts:    Union[List, np.ndarray] = None,
        final_pts:      Union[List, np.ndarray] = None,
        overlay_pts:    Union[List, np.ndarray] = None,
        font_size:      float = 15.,
        colormap:       Union[dict, str, mpl.colors.LinearSegmentedColormap] = None,
        title:          str = None,
        overlay_marker: str = "x",
        overlay_color:  str = "k",
        overlay_size:   float = 20., #None,
        overlay_colors: Union[list, str] = "k",
        transparency:   bool = True,
        equal_aspect:   bool = False,
        num_ticks:      int = None,
        box_xbounds:    tuple = None,
        box_ybounds:    tuple = None,
        box_color:      Union[dict, str, mpl.colors.LinearSegmentedColormap] = "k",
        box_linestyle:  Union[str, tuple] = "-",
        box_alpha:      float = 1,
        basin_alpha:    float = 0.75,
        fp_linewidth:   float = 2.,
        skip_heatmap:   bool = False,
        label_axes:     bool = True,
        xlims:          tuple = None,
        ylims:          tuple = None
        ):
    
    if library is not None:
        logging.warning("Using library data. Ignoring initial_pts and final_pts, if provided.")
        """
        if library.standardize:
            '''
            initial_pts = np.array(library.parameters) + library.standardizer.shift[:2]
            initial_pts *= library.standardizer.scale[:2]
            final_pts = np.array(library.final_parameters) + library.standardizer.shift[:2]
            final_pts *= library.standardizer.scale[:2]
            '''
            initial_pts = np.array(library.parameters) / library.standardizer.scale[:2]
            initial_pts -= library.standardizer.shift[:2]
            final_pts = np.array(library.final_parameters) / library.standardizer.scale[:2]
            final_pts -= library.standardizer.shift[:2]
        else:
        """
        initial_pts = np.array(library.parameters)
        final_pts = np.array(library.final_parameters)
        if get_fixed_pt is None and hasattr(library, "get_basin"):
            get_fixed_pt = library.get_basin
            
    elif initial_pts is None or final_pts is None:
        msg = "No library provided. Please supply both initial_pts and final_pts."
        logging.error(msg)
        raise(NotImplementedError(msg))
    else:
        if isinstance(initial_pts, list):
            initial_pts = np.array(initial_pts)
        if isinstance(final_pts, list):
            final_pts = np.array(final_pts)
    
    if colormap is None:
        #colors_map = ["magenta", "yellow", "black"]
        colors_map = ["firebrick", "darkslategray", "tab:orange"] #, np.nan: "black"}
        colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
        colormap.set_bad("white")
    
    grid_width = int(np.sqrt(len(initial_pts)))

    with mpl.rc_context({'font.size': font_size}):
        figure, _ = plt.subplots(figsize = (6.5, 6), constrained_layout = True,
                                 sharex = True, sharey = True)
        
        ax = plt.subplot2grid(shape = (1, 1), loc = (0, 0), colspan = 1, fig = figure)
            
        mesh_args = {
            "shading" : "nearest", "cmap" : colormap,
            "norm" : mpl.colors.Normalize(clip = False, vmin = 0, vmax = len(fixed_points))
            }
            
        x = initial_pts[:, 0]
        idx = np.unique(x, return_index = True)[1]
        x = np.array([x[index] for index in sorted(idx)])
        
        y = initial_pts[:, 1]
        idy = np.unique(y, return_index = True)[1]
        y = np.array([y[index] for index in sorted(idy)])
        
        x, y = np.meshgrid(x, y, indexing = "ij")
        
        finals = np.array([get_fixed_pt(final) for final in final_pts]).reshape(
            (grid_width, grid_width))
        
        if not skip_heatmap:
            ax.pcolormesh(x, y, finals, **mesh_args, alpha = basin_alpha)
         
        if label_axes:
            ax.set_xlabel("$x_0$")
            ax.set_ylabel("$y_0$")
        
        if title is not None:
            ax.set_title(title)
            
        ax.scatter(fixed_points[:,0], fixed_points[:,1],
                   marker = "x",
                   s = 50.,
                   linewidths = fp_linewidth,
                   #s = 20.,
                   c = "k"
                   )
        
        if overlay_pts is not None:
            overlay_pts = np.array(overlay_pts)
            ax.scatter(overlay_pts[:,0], overlay_pts[:,1],
                       marker = overlay_marker, s = overlay_size,
                       c = overlay_colors)
            
        if equal_aspect:
            ax.set_aspect('equal')
        
        if box_xbounds is not None and box_ybounds is not None:
            ax.vlines(x = box_xbounds[0], ymin = box_ybounds[0],  ymax = box_ybounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.vlines(x = box_xbounds[1], ymin = box_ybounds[0],  ymax = box_ybounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.hlines(y = box_ybounds[0], xmin = box_xbounds[0],  xmax = box_xbounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.hlines(y = box_ybounds[1], xmin = box_xbounds[0],  xmax = box_xbounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            
        if num_ticks is not None:
            #ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
            #ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
            ax.set_xticks(np.linspace(min(initial_pts[:, 0]), max(initial_pts[:, 0]), num_ticks))
            ax.set_yticks(np.linspace(min(initial_pts[:, 1]), max(initial_pts[:, 1]), num_ticks))
        
        if xlims is not None:
            ax.set_xlim(*xlims)
        if ylims is not None:
            ax.set_ylim(*ylims)
        
    if transparency:
        figure.patch.set_alpha(0)
            
def basin_error_heatmap(
        fixed_points:       np.ndarray,
        get_fixed_pt:       Callable,
        initial_pts:        Union[List, np.ndarray],
        predicted_finals:   Union[List, np.ndarray],
        true_finals:        Union[List, np.ndarray],
        font_size:          float = 15.,
        colormap:           Union[dict, str, mpl.colors.LinearSegmentedColormap] = None,
        title:              str = None,
        overlay_pts:        Union[List, np.ndarray] = None,
        overlay_marker:     str = "x",
        overlay_color:      str = "k",
        overlay_size:       float = 20., #None,
        transparency:       bool = True,
        equal_aspect:       bool = False,
        num_ticks:          int = None,
        box_xbounds:        tuple = None,
        box_ybounds:        tuple = None,
        box_color:          Union[dict, str, mpl.colors.LinearSegmentedColormap] = "k",
        box_linestyle:      Union[str, tuple] = "-",
        box_alpha:          float = 1,
        basin_alpha:        float = 0.75,
        fp_linewidth:       float = 2.
        ):
    
    if isinstance(initial_pts, list):
        initial_pts = np.array(initial_pts)
    if isinstance(predicted_finals, list):
        predicted_finals = np.array(predicted_finals)
    if isinstance(true_finals, list):
        true_finals = np.array(true_finals)
        
    #print(predicted_finals)
    '''
    if colormap is None:
        #colors_map = ["magenta", "yellow", "black"]
        colors_map = ["firebrick", "darkslategray", "tab:orange"] #, np.nan: "black"}
        colormap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors_map, N = len(colors_map))
        colormap.set_bad("white")
    '''
    grid_width = int(np.sqrt(len(initial_pts)))

    with mpl.rc_context({'font.size': font_size}):  
        #figure, _ = plt.subplots(figsize = (14, 8), constrained_layout = True,
        #                      sharex = True, sharey = True)
        figure, _ = plt.subplots(figsize = (6.5, 6), constrained_layout = True,
                                 sharex = True, sharey = True)
        
        ax = plt.subplot2grid(shape = (1, 1), loc = (0, 0), colspan = 1, fig = figure)
            
        mesh_args = {
            "shading" : "nearest", "cmap" : colormap,
            "norm" : mpl.colors.Normalize(clip = False, vmin = 0, vmax = len(fixed_points))
            }
            
        x = initial_pts[:, 0]
        idx = np.unique(x, return_index = True)[1]
        x = np.array([x[index] for index in sorted(idx)])
        
        y = initial_pts[:, 1]
        idy = np.unique(y, return_index = True)[1]
        y = np.array([y[index] for index in sorted(idy)])
        
        x, y = np.meshgrid(x, y, indexing = "ij")
        
        predictions = np.array([get_fixed_pt(final) for final in predicted_finals],
                               dtype = float).reshape((grid_width, grid_width))
        truths = np.array([get_fixed_pt(final) for final in true_finals],
                          dtype = float).reshape((grid_width, grid_width))
        errors = abs(predictions - truths)
        
        finals = np.copy(predictions)
        
        condition1 = errors != 0
        condition2 = ~np.isnan(predictions) # != np.nan #-np.inf
        finals[np.logical_and(condition1, condition2)] = 10**15
        finals[np.isnan(predictions)] = -10**15
        #print(np.array(np.nonzero(finals == 10**15)).shape)
        #print(np.array(np.nonzero(finals == -10**15)).shape)
        
        ax.pcolormesh(x, y, finals, **mesh_args, alpha = basin_alpha)
        #figure.colorbar(pcm, ax = ax, extend = "both")
            
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$y_0$")
        
        if title is not None:
            ax.set_title(title)
            
        ax.scatter(fixed_points[:,0], fixed_points[:,1], marker = "x",
                   s = 50., #100.,
                   linewidths = fp_linewidth, #2, #3,
                   #s = 20.,
                   c = "k")#"white")
        
        if overlay_pts is not None:
            overlay_pts = np.array(overlay_pts)
            ax.scatter(overlay_pts[:,0], overlay_pts[:,1],
                       marker = overlay_marker, s = overlay_size, #20.,
                       c = overlay_color) #"k") #"tab:grey")
            
        if equal_aspect:
            ax.set_aspect('equal')
        
        if box_xbounds is not None and box_ybounds is not None:
            ax.vlines(x = box_xbounds[0], ymin = box_ybounds[0],  ymax = box_ybounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.vlines(x = box_xbounds[1], ymin = box_ybounds[0],  ymax = box_ybounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.hlines(y = box_ybounds[0], xmin = box_xbounds[0],  xmax = box_xbounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
            ax.hlines(y = box_ybounds[1], xmin = box_xbounds[0],  xmax = box_xbounds[1],
                      color = box_color, linestyle = box_linestyle, alpha = box_alpha)
        
        if num_ticks is not None:
            #ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
            #ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
            ax.set_xticks(np.linspace(min(initial_pts[:, 0]), max(initial_pts[:, 0]), num_ticks))
            ax.set_yticks(np.linspace(min(initial_pts[:, 1]), max(initial_pts[:, 1]), num_ticks))
    
    if transparency:
        figure.patch.set_alpha(0)
            
def plot_energies(
        trajectory:             Union[basin_prediction, np.ndarray],
        energy_func:            Callable,
        kinetic_func:           Callable = None,
        potential_func:         Callable = None,
        energy_barrier_loc:     np.ndarray = None,
        energy_args:            dict = {},
        ax:                     Union[np.ndarray, mpl.axes._axes.Axes] = None,
        breakdown_energies:     bool = True
        ):
    
    if isinstance(ax, np.ndarray):
        ax = ax[0]
    elif ax is None:
        fig, ax = plt.subplots(constrained_layout = True)
    
    if isinstance(trajectory, np.ndarray):
        with mpl.rc_context({"font.size" : 15}):
            total = energy_func(trajectory, **energy_args)
            potential = potential_func(trajectory, **energy_args)
            kinetic = kinetic_func(trajectory, **energy_args)
            ax.plot(total, label = "Total")
            if breakdown_energies:
                ax.plot(potential, label = "Potential")
                ax.plot(kinetic, label = "Kinetic")
            ax.axhline(y = energy_func(energy_barrier_loc), linestyle = "--", label = "Convergence Threshold")
            ax.set_ylabel("Energy")
            ax.set_xlabel("Time ($\Delta t$)")
            ax.set_xlim(0)
            ax.set_ylim(0)
            ax.legend(frameon = False)
    
    else:
        with mpl.rc_context({"font.size" : 15}):
            if trajectory.resync_inputs is not None:
                truth = np.concatenate((trajectory.resync_inputs,
                                        trajectory.target_outputs))
                resync_time = trajectory.resync_inputs.shape[0]
                ax.axvline(x = resync_time, label = "Prediction Start",
                           linestyle = "dotted", color = "black")                
            else:
                resync_time = 0
                truth = trajectory.target_outputs
            ax.axhline(y = energy_func(energy_barrier_loc, **energy_args), label = "Convergence Threshold",
                       linestyle = "dotted", color = "tab:red")
            full_time = truth.shape[0]
            if breakdown_energies:
                ax.plot(potential_func(truth, **energy_args), label = "Potential",
                        linestyle = "-", color = "tab:orange")
                ax.plot(np.arange(resync_time, full_time),
                        potential_func(trajectory.reservoir_outputs, **energy_args),
                        linestyle = "--", color = "tab:orange")
                ax.plot(kinetic_func(truth, **energy_args), label = "Kinetic",
                        linestyle = "-", color = "tab:green")
                ax.plot(np.arange(resync_time, full_time),
                        kinetic_func(trajectory.reservoir_outputs, **energy_args),
                        linestyle = "--", color = "tab:green")
            ax.plot(energy_func(truth, **energy_args), label = "Total",
                    linestyle = "-", color = "tab:blue")
            ax.plot(np.arange(resync_time, full_time),
                    energy_func(trajectory.reservoir_outputs, **energy_args),
                    linestyle = "--", color = "tab:blue")
            ax.set_ylabel("Energy")
            ax.set_xlabel("Time ($\Delta t$)")
            ax.set_xlim(0)
            ax.set_ylim(0)
            ax.legend(frameon = False)

def plot_prediction_and_energies(
        basin_prediction:       basin_prediction,
        fixed_points:           Union[list, np.ndarray] = None,
        energy_func:            Callable = None,
        kinetic_func:           Callable = None,
        potential_func:         Callable = None,
        energy_barrier_loc:     np.ndarray = None,
        energy_args:            dict = {},
        phase_xlims:            tuple = None,
        phase_ylims:            tuple = None,
        plot_dims:              Union[int, List[int]] = None,
        max_horizon:            int = None,
        frame_legend:           bool = False,
        legend_loc:             tuple = (.5, 1.1), # -.75),
        n_legend_cols:          int = 4,
        font_size:              float = 15.,
        show_plot:              bool = True,
        save_plot:              bool = False,
        save_loc:               str = None,
        save_name:              str = "Pred_and_Energies.png",
        title:                  str = None,
        breakdown_energies:     bool = True
        ):
    
    if basin_prediction.prediction is None:
        msg = "basin_prediction has no stored trajectory, cannot plot."
        logging.warning(msg)
        print(msg)
        
    else:
        forecast = basin_prediction.prediction.reservoir_outputs
        truth = basin_prediction.prediction.target_outputs
        f_resyncs = basin_prediction.prediction.resync_outputs
        t_resyncs = basin_prediction.prediction.resync_inputs
        if max_horizon is not None and max_horizon < forecast.shape[0]:
            forecast = forecast[: max_horizon]
            truth = truth[: max_horizon]
        x = forecast[:, 0]
        y = forecast[:, 1]
        truth_x = truth[:, 0]
        truth_y = truth[:, 1]
        colors = [(c, 0, 1-c) for c in np.linspace(0, 1, forecast.shape[0])]
        truth_colors = [(1, c, 1-c) for c in np.linspace(0, 1, truth.shape[0])]
        
        p_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors, N = len(colors))
        p_cmap.set_bad()
        t_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", truth_colors, N = len(truth_colors))
        t_cmap.set_bad()
        
        with mpl.rc_context({"font.size" : font_size}):
            if np.all(np.var(basin_prediction.prediction.target_outputs, axis = 0)):
                if energy_func is not None:
                    fig = plt.figure(figsize = (19, 6.5), constrained_layout = True) #, squeeze = False)
                else:
                    fig = plt.figure(figsize = (15, 8), constrained_layout = True)#, squeeze = False)
                if energy_func is None:
                    spec = fig.add_gridspec(forecast.shape[1], 2)
                else:
                    spec = fig.add_gridspec(forecast.shape[1], 3)
                ax0 = fig.add_subplot(spec[:, 1])
                ax = [ax0]
                if energy_func is not None:
                    ax1 = fig.add_subplot(spec[:, 2])
                    ax = [ax0, ax1]
                for i in range(forecast.shape[1]):
                    ax.append(fig.add_subplot(spec[i, 0]))
                #ax10 = fig.add_subplot(spec[0, 0])
                #ax11 = fig.add_subplot(spec[1, 0])#, sharex = ax10)
                #ax12 = fig.add_subplot(spec[2, 0])#, sharex = ax10)
                #ax13 = fig.add_subplot(spec[3, 0])#, sharex = ax10)
                #ax = np.array([ax0, ax10, ax11, ax12, ax13]).flatten()
                ax = np.array(ax).flatten()
                cbar_locs = ["top", "right"]
            else:
                fig = plt.figure(figsize = (15, 8), constrained_layout = True)#, squeeze = False)
                spec = fig.add_gridspec(1, 2)
                ax0 = fig.add_subplot(spec[0, 0])
                ax1 = fig.add_subplot(spec[0, 1])
                ax = np.array([ax0, ax1]).flatten()
                cbar_locs = ["top", "right"]
                
            #ax[0].scatter(x = f_resyncs[:, 0], y = f_resyncs[:, 1], c = "k", marker = "o", s = 1.)
            ax[0].scatter(x = t_resyncs[:, 0], y = t_resyncs[:, 1], c = "k", marker = "*", s = 1.)            
            p_cm = ax[0].scatter(x = x, y = y,
                                 c = np.arange(len(forecast)), cmap = p_cmap, #c = colors,
                                 marker = "o", s = 1.)
            t_cm = ax[0].scatter(x = truth_x, y = truth_y,
                                 c = np.arange(len(truth)), cmap = t_cmap, #c = truth_colors,
                                 marker = "*", s = 1.)
            if fixed_points is not None:
                ax[0].scatter(x = fixed_points[:, 0], y = fixed_points[:, 1],
                              c = "k", marker = "o", s = 5., label = "Magnets")
            #ax.plot(x, y, c = colors, marker = "o")
            #ax.plot(truth_x, truth_y, c = truth_colors, marker = "*")
            ax[0].set_xlabel("$x$")
            ax[0].set_ylabel("$y$")
            ax[0].set_xlim(phase_xlims) #-1.5, 1.5)
            ax[0].set_ylim(phase_ylims) #-1.5, 1.5)
            ax[0].set_aspect("equal")
            #ax.legend(legend_loc, frameon = frame_legend)
            p_cbar = fig.colorbar(p_cm, ax = ax[0], label = "Time ($\Delta t$), Prediction",
                                  location = cbar_locs[0], pad = -.05) # -.1)
            t_cbar = fig.colorbar(t_cm, ax = ax[0], label = "Time ($\Delta t$), Truth",
                                  location = cbar_locs[1], pad = 0.) #shrink = .8)
            
            if energy_func is not None:
                plot_energies(
                    trajectory = basin_prediction.prediction,
                    ax = ax[1],
                    breakdown_energies = breakdown_energies,
                    energy_func = energy_func,
                    potential_func = potential_func,
                    kinetic_func = kinetic_func,
                    energy_args = energy_args,
                    energy_barrier_loc = energy_barrier_loc
                    )
            
            if np.all(np.var(basin_prediction.prediction.target_outputs, axis = 0)):
                rch.plot_predict_result(
                    prediction = basin_prediction.prediction,
                    plot_dims = plot_dims,
                    max_horizon = max_horizon,
                    frame_legend = frame_legend,
                    legend_loc = legend_loc,
                    legend_ax = 0,
                    n_legend_cols = 3, #n_legend_cols//2,
                    font_size = font_size,
                    incl_tvalid = False,
                    axes = ax[-forecast.shape[1]:],
                    fig = fig
                    )
                
            if title is not None:
                fig.suptitle(title)
            
            if save_plot:
                if save_loc is None:
                    save_loc = os.getcwd()
                if not os.path.isdir(save_loc):
                    os.makedirs(save_loc)
                file = os.path.join(save_loc, save_name)
                fig.savefig(file)
                if not show_plot:
                    plt.close(fig)
                print("Saved")
                
            if show_plot:
                plt.show()

def plot_state_space(
        train_trajs:            Union[list, np.ndarray] = None,
        resync_trajs:           Union[list, np.ndarray] = None,
        pred_trajs:             Union[list, np.ndarray] = None,
        true_trajs:             Union[list, np.ndarray] = None,
        train_color:            Union[str, List[str]] = "tab:red",
        resync_color:           Union[str, List[str]] = "tab:pink",
        pred_color:             Union[str, List[str]] = "tab:gray",
        true_color:             Union[str, List[str]] = "tab:blue",
        alpha:                  float = 1.,
        fixed_points:           Union[list, np.ndarray] = None,
        phase_xlims:            tuple = None,
        phase_ylims:            tuple = None,
        equal_aspect:           bool = False,
        plot_dims:              Union[int, List[int]] = [0, 1],
        figsize:                tuple = (6.5, 6),
        fig_legend:             bool = False,
        legend:                 bool = True,
        frame_legend:           bool = False,
        legend_loc:             tuple = (.5, 1.1), # -.75),
        n_legend_cols:          int = 4,
        font_size:              float = 15.,
        show_plot:              bool = True,
        save_plot:              bool = False,
        save_loc:               str = None,
        save_name:              str = "Pred_and_Energies.png",
        title:                  str = None,
        transparency:           bool = True,
        num_ticks:              int = None
        ):
    
    if train_trajs is None:
        train_trajs = []
    elif isinstance(train_trajs, np.ndarray):
        train_trajs = [train_trajs]
        
    if resync_trajs is None:
        resync_trajs = []
    elif isinstance(resync_trajs, np.ndarray):
        resync_trajs = [resync_trajs]
        
    if pred_trajs is None:
        pred_trajs = []
    elif isinstance(pred_trajs, np.ndarray):
        pred_trajs = [pred_trajs]
    
    if true_trajs is None:
        true_trajs = []
    elif isinstance(true_trajs, np.ndarray):
        true_trajs = [true_trajs]
        
    if isinstance(train_color, str):
        train_color = [train_color] * len(train_trajs)
        
    if isinstance(pred_color, str):
        pred_color = [pred_color] * len(pred_trajs)
        
    if isinstance(resync_color, str):
        resync_color = [resync_color] * len(resync_trajs)
        
    if isinstance(true_color, str):
        true_color = [true_color] * len(true_trajs)
    
    with mpl.rc_context({"font.size" : font_size}):
        fig, ax = plt.subplots(figsize = figsize, constrained_layout = True)#, squeeze = False)
            
        for i, traj in enumerate(train_trajs):
            if i == 0:
                label = 'Training Trajectory'
            else:
                label = None
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], c = train_color[i],
                    label = label, alpha = alpha)
            ax.scatter(traj[0, plot_dims[0]], traj[0, plot_dims[1]], c = train_color[i],
                       s = 2.)
        
        for i, traj in enumerate(true_trajs):
            if i == 0:
                label = 'True Trajectory'
            else:
                label = None
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], c = true_color[i],
                    label = label, alpha = alpha)
            ax.scatter(traj[0, plot_dims[0]], traj[0, plot_dims[1]], c = true_color[i],
                       s = 2.)
        
        for i, traj in enumerate(resync_trajs):
            if i == 0:
                label = 'Resync Trajectory'
            else:
                label = None
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], c = resync_color[i],
                    label = label, alpha = alpha, linewidth = 3.)
            ax.scatter(traj[0, plot_dims[0]], traj[0, plot_dims[1]], c = resync_color[i],
                       s = 5.) #2.)
        
        for i, traj in enumerate(pred_trajs):
            if i == 0:
                label = 'Predicted Trajectory'
            else:
                label = None
            if len(resync_trajs) == len(pred_trajs):
                traj = np.concatenate((resync_trajs[i][-1][None], traj))
            ax.plot(traj[:, plot_dims[0]], traj[:, plot_dims[1]], c = pred_color[i],
                    label = label, alpha = alpha)
            #ax.scatter(traj[0, plot_dims[0]], traj[0, plot_dims[1]], c = pred_color)
        
        if fixed_points is not None:
            ax.scatter(x = fixed_points[:, 0], y = fixed_points[:, 1],
                       c = "k", marker = "x", s = 50., #100.,
                       linewidths = 2, #3,
                       label = "Fixed Points",
                       zorder = 3
                       )
        
        if fig_legend:
            fig.legend(frameon = frame_legend, loc = legend_loc, ncols = n_legend_cols)
        elif legend:
            ax.legend(frameon = frame_legend, loc = legend_loc, ncols = n_legend_cols)
            
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(phase_xlims)
        ax.set_ylim(phase_ylims)
        if equal_aspect:
            ax.set_aspect("equal")  
            
        if num_ticks is not None:
            ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
            ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))
            #ax.set_xticks(np.linspace(min(initial_pts[:, 0]), max(initial_pts[:, 0]), num_ticks))
            #ax.set_yticks(np.linspace(min(initial_pts[:, 1]), max(initial_pts[:, 1]), num_ticks))
            
        if title is not None:
            fig.suptitle(title)
            
        if transparency:
            fig.patch.set_alpha(0)
        
        if save_plot:
            if save_loc is None:
                save_loc = os.getcwd()
            if not os.path.isdir(save_loc):
                os.makedirs(save_loc)
            file = os.path.join(save_loc, save_name)
            fig.savefig(file)
            if not show_plot:
                plt.close(fig)
            print("Saved")
            
        if show_plot:
            plt.show()
