import numpy as np 
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from typing import Union, Tuple
from emukit.core.acquisition import Acquisition, IntegratedHyperParameterAcquisition
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import AcquisitionOptimizerBase
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound, MaxValueEntropySearch
import scipy.stats
import GPy
from GPy.models import GPRegression
import matplotlib.pyplot as plt


def get_array_and_len(min, max, step):
    """ Get array and its length for process condition options. """
    var = np.arange(min, max + step, step)
    num = len(var)

    return var, num


def x_scaler(X, var_array):  
    """ Min-max scale array X. """
    def max_min_scaler(x, x_max, x_min):
        return (x-x_min)/(x_max-x_min)

    x_norm = []
    for x in (X):
        x_norm.append([max_min_scaler(x[i], 
                                     max(var_array[i]), 
                                     min(var_array[i])) for i in range(len(x))])  
    return np.array(x_norm)


def x_descaler(x_norm, var_array):
    """ Rescale X to original range. """
    def max_min_rescaler(x, x_max, x_min):
        return x*(x_max-x_min)+x_min
    x_original = []
    for x in (x_norm):
           x_original.append([max_min_rescaler(x[i],
                                        max(var_array[i]),
                                        min(var_array[i])) for i in range(len(x))])
    return np.array(x_original)


def get_closest_array(suggested_x, var_array):
    """ Find the array of closest grid points to the points X suggested by the model. """
    def get_closest_value(given_value, array_list):
        absolute_difference_function = lambda list_value : abs(list_value - given_value)
        closest_value = min(array_list, key=absolute_difference_function)
        return closest_value

    var_list = var_array
    modified_array = []
    for x in suggested_x:
        print(x)
        modified_array.append([get_closest_value(x[i], var_list[i]) for i in range(len(x))])
    print(modified_array)
    return np.array(modified_array)


def define_parameter_space(NMP_num, DMF_num, DMSO_num, PC_num, AT_num, VP_num, VT_num, T_num, humidity_num):
    """ Define the parameter space for the model. """
    return ParameterSpace([ContinuousParameter('NMP', 0 - 1 / (NMP_num - 1) / 2, 1 + 1 / (NMP_num - 1) / 2),
                                ContinuousParameter('DMF', 0 - 1 / (DMF_num - 1) / 2, 1 + 1 / (DMF_num - 1) / 2),
                                ContinuousParameter('DMSO', 0 - 1 / (DMSO_num - 1) / 2, 1 + 1 / (DMSO_num - 1) / 2),
                                ContinuousParameter('PC', 0 - 1 / (PC_num - 1) / 2, 1 + 1 / (PC_num - 1) / 2),
                                ContinuousParameter('AT', 0 - 1 / (AT_num - 1) / 2, 1 + 1 / (AT_num - 1) / 2),
                                ContinuousParameter('VP', 0 - 1 / (VP_num - 1) / 2, 1 + 1 / (VP_num - 1) / 2),
                                ContinuousParameter('VT', 0 - 1 / (VT_num - 1) / 2, 1 + 1 / (VT_num - 1) / 2),
                                ContinuousParameter('T', 0 - 1 / (T_num - 1) / 2, 1 + 1 / (T_num - 1) / 2),
                                ContinuousParameter('humidity', 0 - 1 / (humidity_num - 1) / 2, 1 + 1 / (humidity_num - 1) / 2),
                                ])


def get_matern52_kernel(input_dim):
    """ Return an instance of the matern52 kernel. """
    ker = GPy.kern.Matern52(input_dim = input_dim, ARD =True)
    ker.lengthscale.constrain_bounded(1e-1, 1)
    ker.variance.constrain_bounded(1e-1, 1000.0)

    return ker


def get_rbf_kernel(input_dim):
    """ Return an instance of the RBF kernel. """
    ker = GPy.kern.RBF(input_dim = input_dim, ARD = True)
    ker.lengthscale.constrain_bounded(1e-1, 1)#upper bound set to 1 
    ker.variance.constrain_bounded(1e-1, 1000.0) 

    return ker


def get_gpr_model(X, Y, ker, set_noise=False):
    """ Build a GPR model based on an input array X, a target array Y, and a kernel ker. """
    model_gpy = GPRegression(X , -Y, ker)#Emukit is a minimization tool; need to make Y negative
    if set_noise:
        model_gpy.Gaussian_noise.variance = 1.5**2
        model_gpy.Gaussian_noise.variance.fix()
    model_gpy.randomize()
    model_gpy.optimize_restarts(num_restarts=20,verbose=False, messages=False)

    return model_gpy


def generate_visualization_suggested_process_conditions(df, n_col, var_array):
    """ Generate bar plots showing the frequency of each process condition in the new sampling suggestions. """
    df_cols = df.columns
    for n in np.arange(0, 8, n_col):
        fig,axes = plt.subplots(1, n_col, figsize=(18, 3.5), sharey = False)
        fs = 20
        for i in np.arange(n_col):
            if n< len(df_cols):
                axes[i].hist(df.iloc[:,n], bins= 20, range = (min(var_array[n]),max(var_array[n])))####
                axes[i].set_xlabel(df_cols[n], fontsize = 18)
            else:
                axes[i].axis("off")
            n = n+1      
        axes[0].set_ylabel('counts', fontsize = 18)
        for i in range(len(axes)):
            axes[i].tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
            axes[i].grid(True, linestyle='-.')
        plt.show()


def generate_visualization_efficiency_vs_ml_conditions(X_new, Xc, df_device, df_film, f_obj, acq_fcn, acq_cons, acq_produc, var_array):
    """ Generate plots of efficiency vs. machine learning conditions."""
    device_eff = df_device.sort_values('Condition').iloc[:,[0,-2]].values
    film_quality = df_film.sort_values('Condition').iloc[:,[0,-1]].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey = False)
    fs = 20
    exp_cond = np.transpose(device_eff)[0]
    exp_eff = np.transpose(device_eff)[1]

    axes[0].scatter(exp_cond, exp_eff, #facecolor = 'none',
            edgecolor = 'navy', s = 20, alpha = 0.6, label = 'experiment')

    unsuccess_film = np.transpose(film_quality[film_quality[:,-1] ==0])
    all_cond = np.concatenate([device_eff, np.transpose(unsuccess_film)])
    all_cond = all_cond[np.argsort(all_cond[:,0])]
    axes[0].plot(np.transpose(all_cond)[0], np.maximum.accumulate(np.transpose(all_cond)[1]), 
         marker = 'o', ms = 0, c = 'black')

    X_sorted = x_scaler(df_film.sort_values('Condition').iloc[:,1:10].values, var_array)
    y_pred, y_uncer = f_obj(X_sorted)
    y_pred = -y_pred[:,-1]
    y_uncer = np.sqrt(y_uncer[:,-1])

    axes[0].scatter(np.arange(len(X_sorted)), y_pred,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
    axes[0].errorbar(np.arange(len(X_sorted)), y_pred, yerr = y_uncer,  
                ms = 0, ls = '', capsize = 2, alpha = 0.6, color = 'gray', zorder = 0)


    y_pred_new, y_uncer_new = f_obj(X_new)
    y_pred_new = -y_pred_new[:,-1]
    y_uncer_new = np.sqrt(y_uncer_new[:,-1])

    axes[0].scatter(np.arange(len(X_new))+len(Xc), y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'darkgreen', label = 'suggested')
    axes[0].errorbar(np.arange(len(X_new))+len(Xc), y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'darkgreen', zorder = 0)


    axes[0].set_ylabel('Current Best Efficiency', fontsize = 20)
    axes[0].set_xlabel('Process Condition', fontsize = 20)

    axes[0].set_ylim(-1, 25)
    axes[0].set_xlim(-1, 45)
    axes[0].set_xticks(np.arange(0,41,10))
    axes[0].legend(fontsize = fs*0.7)

    axes[1].plot(np.arange(len(X_new))+len(Xc), acq_cons, marker = 'o',
                ms = 2, alpha = 0.6, color = 'red', label = 'constr prob')
    axes[1].plot(np.arange(len(X_new))+len(Xc), acq_fcn/20, marker = 'o',
                ms = 2, alpha = 0.6, color = 'navy', label = 'raw acqui')

    axes[1].plot(np.arange(len(X_new))+len(Xc), acq_produc/20, marker = 'o',
                ms = 2, alpha = 0.6, color = 'darkgreen', label = 'final acqui')


    axes[1].set_ylim(0.0, 2)
    axes[1].set_xlim(-1, 45)
    axes[1].set_xticks(np.arange(0,45,10))
    axes[1].set_ylabel('Acquisition Probability', fontsize = fs)
    axes[1].set_xlabel('Process Condition', fontsize = fs)

    for ax in axes:
        ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
        ax.grid(True, linestyle='-.')
    plt.subplots_adjust(wspace = 0.4)
    plt.legend(fontsize = fs*0.7)
    plt.show()


def generate_contour_plot(ind1, ind2, x_sampled, f_obj, x_descaler, x_columns, var_array):
    n_steps = 21
    x1x2y_pred, x1x2y_uncer = [[], []]
    for x1 in np.linspace(0, 1, n_steps):
        for x2 in np.linspace(0, 1, n_steps):
            x_temp = np.copy(x_sampled)
            x_temp[:, ind1] = x1
            x_temp[:, ind2] = x2
            y_pred, y_uncer = f_obj(x_temp)
            y_pred = -y_pred
            x1_org = x_descaler(x_temp, var_array)[0, ind1]
            x2_org = x_descaler(x_temp, var_array)[0, ind2]
            x1x2y_pred.append([x1_org, x2_org, np.max(y_pred), np.mean(y_pred), np.min(y_pred)])
            x1x2y_uncer.append([x1_org, x2_org, np.max(np.sqrt(y_uncer)), np.mean(np.sqrt(y_uncer)), np.min(np.sqrt(y_uncer))])

    x1 = np.array(x1x2y_pred, dtype=object)[:, 0].reshape(n_steps, n_steps)
    x2 = np.array(x1x2y_pred, dtype=object)[:, 1].reshape(n_steps, n_steps)

    y_pred_max = np.array(x1x2y_pred, dtype=object)[:, 2].reshape(n_steps, n_steps)
    y_pred_mean = np.array(x1x2y_pred, dtype=object)[:, 3].reshape(n_steps, n_steps)
    y_pred_min = np.array(x1x2y_pred, dtype=object)[:, 4].reshape(n_steps, n_steps)

    y_uncer_max = np.array(x1x2y_uncer, dtype=object)[:, 2].reshape(n_steps, n_steps)
    y_uncer_mean = np.array(x1x2y_uncer, dtype=object)[:, 3].reshape(n_steps, n_steps)
    y_uncer_min = np.array(x1x2y_uncer, dtype=object)[:, 4].reshape(n_steps, n_steps)

    fs = 20
    title_pad = 16

    # Contour for Prediction Efficiency Mean
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=False, sharex=False)
    fig.subplots_adjust(wspace=0.4, hspace=None)
    colorbar_offset = [12.5, 7, 4]
    for ax, c_offset, y in zip(axes, colorbar_offset,
                               [y_pred_max, y_pred_mean, y_pred_min]):

        c_plt1 = ax.contourf(x1, x2, y, levels=np.arange(19) * 0.25 + c_offset, cmap='plasma', extend = 'both')
        cbar = fig.colorbar(c_plt1, ax=ax)
        cbar.ax.tick_params(labelsize=fs * 0.8)
        ax.set_xlabel(str(x_columns[ind1]), fontsize=fs)
        ax.set_ylabel(str(x_columns[ind2]), fontsize=fs)

        x1_delta = (np.max(x1) - np.min(x1)) * 0.05
        x2_delta = (np.max(x2) - np.min(x2)) * 0.05
        ax.set_xlim(np.min(x1) - x1_delta, np.max(x1) + x1_delta)
        ax.set_ylim(np.min(x2) - x2_delta, np.max(x2) + x2_delta)
        ax.tick_params(direction='in', length=5, width=1, labelsize=fs * 0.8)

    axes[0].set_title('objective fcn max', pad=title_pad, fontsize=fs)
    axes[1].set_title('objective fcn mean', pad=title_pad, fontsize=fs)
    axes[2].set_title('objective fcn min', pad=title_pad, fontsize=fs)


class ProbabilisticConstraintBayesianOptimizationLoop2(OuterLoop):
    def __init__(self, space: ParameterSpace, model_objective: Union[IModel, IDifferentiable],
                 model_constraint1: Union[IModel, IDifferentiable], 
                 model_constraint2: Union[IModel, IDifferentiable],
                 acquisition: Acquisition = None,
                 update_interval: int = 1, batch_size: int = 1):

        """
        Emukit class that implements a loop for building Bayesian optimization with an unknown constraint.
        For more information see:
        Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams,
        Bayesian Optimization with Unknown Constraints,
        https://arxiv.org/pdf/1403.5607.pdf
        :param space: Input space where the optimization is carried out.
        :param model_objective: The model that approximates the underlying objective function
        :param model_constraint: The model that approximates the unknown constraints
        :param acquisition: The acquisition function for the objective function (default, EI).
        :param update_interval:  Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        """

        self.model_objective = model_objective
        self.model_constraint1 = model_constraint1
        self.model_constraint2 = model_constraint2
        
        if acquisition is None:
            acquisition = ExpectedImprovement(model_objective)
        
        acquisition_constraint1 = ScaledProbabilityOfFeasibility(model_constraint1, max_value = 1, min_value = 0.5)
        acquisition_constraint2 = ScaledProbabilityOfFeasibility(model_constraint2, max_value = 1, min_value = 0.8)
        acquisition_constraint = acquisition_constraint1 * acquisition_constraint2
        acquisition_constrained = acquisition * acquisition_constraint

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval)
        model_updater_constraint1 = FixedIntervalUpdater(model_constraint1, update_interval,
                                                        lambda state: state.Y_constraint1)
        model_updater_constraint2 = FixedIntervalUpdater(model_constraint2, update_interval,
                                                        lambda state: state.Y_constraint2)

        acquisition_optimizer = GradientAcquisitionOptimizer(space)

        if batch_size == 1:
            candidate_point_calculator = SequentialPointCalculator(acquisition_constrained, acquisition_optimizer)
        else:
            log_acquisition = LogAcquisition(acquisition_constrained)
            candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer,
                                                                          model_objective, space, batch_size)
        loop_state = create_loop_state(model_objective.X, model_objective.Y)
        
        super(ProbabilisticConstraintBayesianOptimizationLoop2, self).__init__(candidate_point_calculator,
                                                                              [model_updater_objective, model_updater_constraint1,model_updater_constraint2],
                                                                              loop_state)


class ScaledProbabilityOfFeasibility(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float = float(0),
                 max_value: float = float(1), min_value: float = float(0)) -> None:
        """
        This acquisition computes for a given input point the probability of satisfying the constraint
        C<0. For more information see:
        Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams,
        Bayesian Optimization with Unknown Constraints,
        https://arxiv.org/pdf/1403.5607.pdf
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param jitter: Jitter to balance exploration / exploitation
        """
        self.model = model
        self.jitter = jitter
        self.max_value = max_value
        self.min_value = min_value

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the probability of of satisfying the constraint C<0.
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        :return: numpy array with the probability of satisfying the constraint at the points x.
        """
        mean, variance = self.model.predict(x)
        mean += self.jitter

        standard_deviation = np.sqrt(variance)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        return cdf*(self.max_value-self.min_value)+self.min_value

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the  probability of of satisfying the constraint C<0.
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        :return: tuple of numpy arrays with the probability of satisfying the constraint at the points x 
        and its gradient.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_devidation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u = - mean / standard_deviation
        pdf = scipy.stats.norm.pdf(0, mean, standard_deviation)
        cdf = scipy.stats.norm.cdf(0, mean, standard_deviation)
        dcdf_dx = - pdf * (dmean_dx + dstandard_devidation_dx * u)

        return cdf*(self.max_value-self.min_value)+self.min_value, dcdf_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
