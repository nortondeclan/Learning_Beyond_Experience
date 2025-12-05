#%% Import Statements
import numpy as np
from typing import Callable, Literal, Union, Generator, List
import numba

from scipy.integrate import solve_ivp
from numpy.random import default_rng

#%% Multistable_Lorenz
@numba.jit(nopython = True, fastmath = True)
def _mlorenz_deriv(x, sigma, beta, rho, omega):
    
    """
    Returns the derivatives for each component of the Lorenz system.
    """
    
    x_prime = np.zeros((3))
    x_prime[0] = omega * ( - (sigma * beta)/(sigma + beta) * x[0] - x[1] * x[2] + rho)
    x_prime[1] = omega * (sigma * x[1] + x[0] * x[2])
    x_prime[2] = omega * (beta * x[2] + x[0] * x[1])

    return x_prime

@numba.jit(nopython=True, fastmath=True)
def _mlorenz(sigma, beta, rho, omega, x0, integrate_length, h):
    
    """
    Applies Runge-Kutta integration to the Lorenz system.
    """
    
    x = np.zeros((integrate_length, 3))
    x[0] = x0

    for t in range(integrate_length - 1):
        
        k1 = _mlorenz_deriv(x[t], sigma[t], beta[t], rho[t], omega[t])
        k2 = _mlorenz_deriv(x[t] + (h/2)*k1, sigma[t], beta[t], rho[t], omega[t])
        k3 = _mlorenz_deriv(x[t] + (h/2)*k2, sigma[t], beta[t], rho[t], omega[t])
        k4 = _mlorenz_deriv(x[t] + h*k3, sigma[t], beta[t], rho[t], omega[t])

        x[t+1] = x[t] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

def get_multistable_lorenz(
    sigma:            Union[float, np.ndarray]             = -10.0,
    beta:             Union[float, np.ndarray]             = -4.0,
    rho:              Union[float, np.ndarray]             = 18.1,
    omega:            Union[float, np.ndarray]             = 1.,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    return_dims:      Union[int, List[int]]                = [0, 1, 2],
    return_mask:      Callable                             = None,
    return_every:     int                                  = 1
    ) -> np.ndarray:
    
    """The Lorenz-getting function.
    
    This function integrates and returns a solution to the Lorenz system,
    obtained with a Runge-Kutta integration scheme.
    
    Args:
        sigma (float, np.ndarray): The first Lorenz parameter.
        beta (float, np.ndarray): The second Lorenz parameter.
        rho (float, np.ndarray): The third Lorenz parameter.
        omega (float, np.ndarray): The time-scale of the Lorenz system.
        x0 (np.ndarray): The initial condition of the Lorenz system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(sigma, float): sigma = sigma * np.ones(integrate_length)
    if isinstance(beta, float): beta = beta * np.ones(integrate_length)
    if isinstance(rho, float): rho = rho * np.ones(integrate_length)
    if isinstance(omega, float): omega = omega * np.ones(integrate_length)
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)

    # Create a random intiail condition, if applicable.
    if isinstance(x0, str) and x0 == 'random':
        
        # Locations and scales are according to observed means and standard
        # deviations of the Lorenz attractor with default parameters.
        x0_0 = rng.uniform(low = -20, high = 20) #normal(loc=-0.036, scale=8.236)
        x0_1 = rng.uniform(low = -20, high = 20) #normal(loc=-0.036, scale=9.162)
        x0_2 = rng.uniform(low = -20, high = 20) #normal(loc=25.104, scale=7.663)
        x0 = np.array([x0_0, x0_1, x0_2])
        
    # Integrate the Lorenz system and return.    
    x = _mlorenz(sigma, beta, rho, omega, x0, integrate_length, h)
    
    if return_mask is None:
        masked = x[transient_length::return_every, return_dims].copy()
    else:
        masked = return_mask(x[transient_length::return_every].copy())
    
    return masked

#%% Magnetic_Pendulum

def mp_distance(
        mag_loc:        np.ndarray,
        pend_loc:       np.ndarray,
        height:         float
        ):
    
    return np.sqrt(np.sum(np.square(mag_loc - pend_loc), axis = 1) + height**2)

@numba.jit(nopython = True, fastmath = True)
def _MP_distance(
        mag_loc : np.ndarray,
        pend_loc : np.ndarray,
        height : float
        ):
    
    return np.sqrt(np.sum(np.square(mag_loc-pend_loc)) + height**2)

def mp_potential_energy(
        state:      np.ndarray,
        mag_locs:   np.ndarray = np.array([[1./np.sqrt(3), 0.],
                                           [-1./(2.*np.sqrt(3)), -.5],
                                           [-1./(2.*np.sqrt(3)), .5]]),
        height:     float = .2,
        frequency:  float = .5
        ):
    
    zero_point = - np.sum(np.array([
        1 / _MP_distance(mag_loc = mag_loc, pend_loc = mag_locs[0].reshape((1, -1)), height = height)
        for mag_loc in mag_locs]), axis = 0)
   
    return .5 * frequency**2 * np.sum(state[:, :2]**2, axis = 1) - np.sum(np.array([
        1 / _MP_distance(mag_loc = mag_loc, pend_loc = state[:, :2], height = height)
        for mag_loc in mag_locs]), axis = 0) - zero_point

def mp_kinetic_energy(
        state:      np.ndarray,
        mag_locs:   np.ndarray = np.array([[1./np.sqrt(3), 0.],
                                           [-1./(2.*np.sqrt(3)), -.5],
                                           [-1./(2.*np.sqrt(3)), .5]]),
        height:     float = .2,
        frequency:  float = .5
        ):
    
    return .5 * np.sum(state[:, 2:]**2, axis = 1)

def mp_energy(
        state:      np.ndarray,
        mag_locs:   np.ndarray = np.array([[1./np.sqrt(3), 0.],
                                           [-1./(2.*np.sqrt(3)), -.5],
                                           [-1./(2.*np.sqrt(3)), .5]]),
        height:     float = .2,
        frequency:  float = .5
        ):
    
    if len(state.shape) == 1:
        state = state.reshape((1, -1))
    
    return mp_kinetic_energy(state) + mp_potential_energy(state, mag_locs, height, frequency)

@numba.jit(nopython = True, fastmath = True)
def _MP_2nd_Derivs(
        state : np.ndarray,
        mag_locs : np.ndarray,
        height : float,
        frequency : float,
        damping : float
        ):
    
    return np.array([- frequency**2 * state[i] - damping * state[i+2] +
                     np.sum(np.array([
                         (mag_locs[j,i] - state[i])/_MP_distance(mag_locs[j], state[:2], height)**3
                         for j in range(len(mag_locs))
                         ])) for i in range(2)])

def get_magnetic_pendulum(
        height:             float = .2,
        frequency:          float = .5,
        damping:            float = .2,
        time_step:          float = .02,
        initial_state:      Union[List[float], np.ndarray] = [-1.2, .75, 0., 0.],
        transient_length:   int = 0,
        return_length:      int = 4000,
        seed:               int = 10,
        return_dims:        Union[int, List[int], np.ndarray] = np.arange(4),
        mag_locs:           np.ndarray = np.array([[1./np.sqrt(3), 0.],
                                                   [-1./(2.*np.sqrt(3)), -.5],
                                                   [-1./(2.*np.sqrt(3)), .5]]),
        direction:          str = "forward",
        return_every:       int = 1
		):
    
    """
    Returns a trajectory of the magnetic pendulum system (Zhang and Cornelius, 2023),
    using default values from their paper.
    """
    
    def MP_Derivs(
            time : float,
            state : np.ndarray
            ):
            
        derivative = np.zeros(4)
        
        derivative[0] = state[2]
        derivative[1] = state[3]
        double_derivatives = _MP_2nd_Derivs(state, mag_locs, height, frequency, damping)
        derivative[2] = double_derivatives[0]
        derivative[3] = double_derivatives[1]
        
        if direction == "forward":
            return derivative
        elif direction == "backward":
            return -derivative

    # Create the random state for reproducibility.
    rng = default_rng(seed)
    if initial_state is None:
        initial_state = np.array([rng.normal(loc = 0., scale = 1) for dim in range(4)])
    else:
        initial_state = np.array(initial_state)

    eval_times = np.arange(0, (transient_length + return_length) * time_step, time_step)
    time = (0, (transient_length + return_length) * time_step)
    #trajectory = odeint(MP_Derivs, initial_state, eval_time)
    trajectory = solve_ivp(MP_Derivs, time, initial_state,
                           method = 'DOP853',
                           t_eval = eval_times
                           ).y.T
	
    return trajectory[transient_length::return_every, return_dims].copy()


#%% Unforced Duffing System

def _unforced_duffing_potential(x, a, b, c):
    
    zero_point = np.array([[-np.sqrt(-b/c), 0]])
    
    return .5 * b * x[:, 0]**2 + .25 * c * x[:, 0]**4 - \
        (.5 * b * zero_point[:, 0]**2 + .25 * c * zero_point[:, 0]**4)

def _unforced_duffing_kinetic(x, a, b, c):
    
    return .5 * x[:, 1]**2

def unforced_duffing_energy(state, a, b, c):
    
    if len(state.shape) == 1:
        state = state.reshape((1, -1))
    
    return _unforced_duffing_kinetic(state, a, b, c) + _unforced_duffing_potential(state, a, b, c)

@numba.jit(nopython = True, fastmath = True)
def _duffing_deriv(x, a, b, c, noise):
    
    """
    Returns the derivatives for each component of the Lorenz system.
    """
    
    x_prime = np.zeros((2))
    x_prime[0] = x[1] + noise[0]
    x_prime[1] = a * x[1] - x[0] * (b + c * x[0]**2) + noise[1]

    return x_prime

@numba.jit(nopython=True, fastmath=True)
def _unforced_duffing(a, b, c, x0, integrate_length, h, direction, noise):
    
    """
    Applies Runge-Kutta integration to the unforced Duffing system.
    """
    
    x = np.zeros((integrate_length, 2))
    x[0] = x0
    
    if direction == 'forward':
        mult = 1
    elif direction == 'backward':
        mult = -1

    for t in range(integrate_length - 1):
        
        k1 = _duffing_deriv(x[t], a[t], b[t], c[t], noise[t])
        k2 = _duffing_deriv(x[t] + (h/2)*k1, a[t], b[t], c[t], noise[t])
        k3 = _duffing_deriv(x[t] + (h/2)*k2, a[t], b[t], c[t], noise[t])
        k4 = _duffing_deriv(x[t] + h*k3, a[t], b[t], c[t], noise[t])

        x[t+1] = x[t] + mult * (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

def get_unforced_duffing(
    a:                Union[float, np.ndarray]             = -0.5,
    b:                Union[float, np.ndarray]             = -1.,
    c:                Union[float, np.ndarray]             = 0.1,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    return_dims:      Union[int, List[int]]                = [0, 1],
    direction:        str                                  = 'forward',
    return_every:     int                                  = 1,
    process_noise:    float                                = 0.
    ) -> np.ndarray:
    
    """The Duffing-getting function.
    
    This function integrates and returns a solution to the unforced Duffing
    system, obtained with a Runge-Kutta integration scheme.
    
    Args:
        a (float, np.ndarray): The first Duffing parameter.
        b (float, np.ndarray): The second Duffing parameter.
        c (float, np.ndarray): The third Duffing parameter.
        x0 (np.ndarray): The initial condition of the Duffing system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(a, float): a = a * np.ones(integrate_length)
    if isinstance(b, float): b = b * np.ones(integrate_length)
    if isinstance(c, float): c = c * np.ones(integrate_length)
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    noise = rng.normal(loc = 0, scale = process_noise, size = (integrate_length, 2))

    # Create a random intiail condition, if applicable.
    if isinstance(x0, str) and x0 == 'random':
        x0_0 = rng.normal()
        x0_1 = rng.normal()
        x0 = np.array([x0_0, x0_1])

    # Integrate the Lorenz system and return.    
    x = _unforced_duffing(a, b, c, x0, integrate_length, h, direction, noise)
    
    return x[transient_length::return_every, return_dims].copy()

#%% Multi-well System

def _double_well_potential(x, a, b, c, d):
    
    zero_point = np.array([0, 0])
    
    return -0.25 * x[:, 0]**2 * a**4 + .5 * b**2 * x[:, 0]**4 - \
        (-0.25 * zero_point[0]**2 * a**4 + .5 * b**2 * zero_point[0]**4) \
            -0.25 * x[:, 1]**2 * c**4 + .5 * d**2 * x[:, 1]**4 - \
                (-0.25 * zero_point[1]**2 * c**4 + .5 * d**2 * zero_point[1]**4)

def _double_well_kinetic(x, a, b, c, d):
    
    return [.5 * np.sum(_double_well_deriv(x[t], a, b, c, d, np.zeros(2))**2)
            for t in range(x.shape[0])]

def double_well_energy(state, a, b, c, d):
    
    if len(state.shape) == 1:
        state = state.reshape((1, -1))
    
    return _double_well_kinetic(state, a, b, c, d) + _double_well_potential(state, a, b, c, d)

@numba.jit(nopython = True, fastmath = True)
def _double_well_deriv(x, a, b, c, d, noise):
    
    """
    Returns the derivatives for each component of the Lorenz system.
    """
    
    x_prime = np.zeros((2))
    x_prime[0] = 0.5 * a**4 * x[0] - 2 * b**2 * x[0]**3 + noise[0]
    x_prime[1] = 0.5 * c**4 * x[1] - 2 * d**2 * x[1]**3 + noise[1]

    return x_prime

@numba.jit(nopython=True, fastmath=True)
def _double_well(a, b, c, d, x0, integrate_length, h, direction, noise):
    
    """
    Applies Runge-Kutta integration to the unforced Duffing system.
    """
    
    x = np.zeros((integrate_length, 2))
    x[0] = x0
    
    if direction == 'forward':
        mult = 1
    elif direction == 'backward':
        mult = -1

    for t in range(integrate_length - 1):
        
        k1 = _double_well_deriv(x[t], a[t], b[t], c[t], d[t], noise[t])
        k2 = _double_well_deriv(x[t] + (h/2)*k1, a[t], b[t], c[t], d[t], noise[t])
        k3 = _double_well_deriv(x[t] + (h/2)*k2, a[t], b[t], c[t], d[t], noise[t])
        k4 = _double_well_deriv(x[t] + h*k3, a[t], b[t], c[t], d[t], noise[t])

        x[t+1] = x[t] + mult * (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

def get_double_well(
    a:                Union[float, np.ndarray]             = 1.,
    b:                Union[float, np.ndarray]             = 1.,
    c:                Union[float, np.ndarray]             = 1.,
    d:                Union[float, np.ndarray]             = 1.,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    return_dims:      Union[int, List[int]]                = [0, 1],
    direction:        str                                  = 'forward',
    return_time:      bool                                 = False,
    process_noise:    float                                = 0.,
    return_every:     int                                  = 1
    ) -> np.ndarray:
    
    """The Duffing-getting function.
    
    This function integrates and returns a solution to the unforced Duffing
    system, obtained with a Runge-Kutta integration scheme.
    
    Args:
        a (float, np.ndarray): The first Duffing parameter.
        b (float, np.ndarray): The second Duffing parameter.
        c (float, np.ndarray): The third Duffing parameter.
        x0 (np.ndarray): The initial condition of the Duffing system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(a, float): a = a * np.ones(integrate_length)
    if isinstance(b, float): b = b * np.ones(integrate_length)
    if isinstance(c, float): c = c * np.ones(integrate_length)
    if isinstance(d, float): d = d * np.ones(integrate_length)
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    noise = rng.normal(loc = 0, scale = process_noise, size = (integrate_length, 2))
    
    # Create a random intiail condition, if applicable.
    if isinstance(x0, str) and x0 == 'random':
        x0 = rng.uniform(low = -1, high = 1, size = 2)

    # Integrate the Lorenz system and return.    
    x = _double_well(a, b, c, d, x0, integrate_length, h, direction, noise)
    
    returned = x[transient_length::return_every, return_dims].copy()
    
    if return_time:
        print('timing')
        returned = np.hstack((
            returned, h * np.arange(transient_length, integrate_length, return_every).reshape((-1, 1))
            ))
    
    return returned
#%% Forced Duffing System

def _forced_duffing_potential(x, a, b, c):
    
    zero_point = np.array([[-np.sqrt(-b/c), 0]])
    
    return .5 * b * x[:, 0]**2 + .25 * c * x[:, 0]**4 - \
        (.5 * b * zero_point[:, 0]**2 + .25 * c * zero_point[:, 0]**4)

def _forced_duffing_kinetic(x, a, b, c):
    
    return .5 * x[:, 1]**2

def forced_duffing_energy(state, a, b, c):
    
    if len(state.shape) == 1:
        state = state.reshape((1, -1))
    
    return _forced_duffing_kinetic(state, a, b, c) + _forced_duffing_potential(state, a, b, c)

@numba.jit(nopython = True, fastmath = True)
def _forced_duffing_deriv(x, a, b, c, f0, fv, w, t, noise):
    
    """
    Returns the derivatives for each component of the Lorenz system.
    """
    
    x_prime = np.zeros((2))
    x_prime[0] = x[1] + noise[0]
    x_prime[1] = f0 + fv * np.cos(w * t) + a * x[1] - x[0] * (b + c * x[0]**2) + noise[1]

    return x_prime

@numba.jit(nopython=True, fastmath=True)
def _forced_duffing(a, b, c, f0, fv, w, x0, integrate_length, h, direction, noise):
    
    """
    Applies Runge-Kutta integration to the unforced Duffing system.
    """
    
    x = np.zeros((integrate_length, 2))
    x[0] = x0
    
    if direction == 'forward':
        mult = 1
    elif direction == 'backward':
        mult = -1

    for t in range(integrate_length - 1):
        
        k1 = _forced_duffing_deriv(x[t], a[t], b[t], c[t], f0[t], fv[t], w[t], h * t, noise[t])
        k2 = _forced_duffing_deriv(x[t] + (h/2)*k1, a[t], b[t], c[t], f0[t], fv[t], w[t], h * t, noise[t])
        k3 = _forced_duffing_deriv(x[t] + (h/2)*k2, a[t], b[t], c[t], f0[t], fv[t], w[t], h * t, noise[t])
        k4 = _forced_duffing_deriv(x[t] + h*k3, a[t], b[t], c[t], f0[t], fv[t], w[t], h * t, noise[t])

        x[t+1] = x[t] + mult * (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

def get_forced_duffing(
    a:                Union[float, np.ndarray]             = -0.5,
    b:                Union[float, np.ndarray]             = -1.,
    c:                Union[float, np.ndarray]             = 0.1,
    f0:               Union[float, np.ndarray]             = 0.1,
    fv:               Union[float, np.ndarray]             = 0.1,
    w:                Union[float, np.ndarray]             = 1.,
    x0:               Union[Literal['random'], np.ndarray] = 'random',
    transient_length: int                                  = 5000,
    return_length:    int                                  = 100000,
    h:                float                                = 0.01,
    seed:             Union[int, None, Generator]          = None,
    return_dims:      Union[int, List[int]]                = [0, 1],
    direction:        str                                  = 'forward',
    return_time:      bool                                 = False,
    process_noise:    float                                = 0.,
    return_every:     int                                  = 1
    ) -> np.ndarray:
    
    """The Duffing-getting function.
    
    This function integrates and returns a solution to the unforced Duffing
    system, obtained with a Runge-Kutta integration scheme.
    
    Args:
        a (float, np.ndarray): The first Duffing parameter.
        b (float, np.ndarray): The second Duffing parameter.
        c (float, np.ndarray): The third Duffing parameter.
        x0 (np.ndarray): The initial condition of the Duffing system.
                         Must be an array of floats of shape (3,).
        transient_length (int): The length (in units of h) of the initial
                                transient to be discarded.
        return_length (int): The length (in units of h) of the returned
                             solution.
        h (float): The integration time step for the Euler scheme.
        seed (int): An integer for determining the random seed of the random
                    initial state.
                    Is only used if x0 is 'random'.
        
    Returns:
        x (np.ndarray): The array of floats describing the solution.
                        Has shape (return_length, 3).
    """
    
    integrate_length = transient_length + return_length
    if isinstance(a, float): a = a * np.ones(integrate_length)
    if isinstance(b, float): b = b * np.ones(integrate_length)
    if isinstance(c, float): c = c * np.ones(integrate_length)
    if isinstance(f0, float): f0 = f0 * np.ones(integrate_length)
    if isinstance(fv, float): fv = fv * np.ones(integrate_length)
    if isinstance(w, float): w = w * np.ones(integrate_length)
    
    # Create the random state for reproducibility.
    rng = default_rng(seed)
    
    noise = rng.normal(loc = 0, scale = process_noise, size = (integrate_length, 2))
    
    # Create a random intiail condition, if applicable.
    if isinstance(x0, str) and x0 == 'random':
        x0_0 = rng.uniform(low = -10, high = 10)
        x0_1 = rng.uniform(low = -10, high = 10)
        x0 = np.array([x0_0, x0_1])

    # Integrate the Lorenz system and return.    
    x = _forced_duffing(a, b, c, f0, fv, w, x0, integrate_length, h, direction, noise)
    
    returned = x[transient_length::return_every, return_dims].copy()
    
    if return_time:
        print('timing')
        returned = np.hstack((
            returned, h * np.arange(transient_length, integrate_length, return_every).reshape((-1, 1))
            ))
    
    return returned