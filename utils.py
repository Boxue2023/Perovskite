import numpy as np 
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from typing import Union, Tuple
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
import scipy.stats
import GPy
from GPy.models import GPRegression
import matplotlib.pyplot as plt
import pandas as pd


def compute_group_means(df):
    """ Filter a dataframe so that only a single row is obtained per
    set of conditions and the mean target values are displayed per set.

    Args:
        df (pd.DataFrame): the experimental data

    Returns:
        pd.DataFrame: the filtered dataframe
    """
    # Group the data by the input column and compute mean for target columns
    grouped_df = df.groupby('Condition').agg({'FF(%)': 'mean', 
                                               'Eff(%)': 'mean',
                                              'Jsc(mA/cm^2)': 'mean'}).reset_index()
    
    # Create a new DataFrame with one entry for each class based on the input column
    filtered_df = pd.DataFrame()
    
    # Iterate over unique values in the input column
    for value in df['Condition'].unique():
        # Filter the grouped DataFrame for the current value
        class_df = grouped_df[grouped_df['Condition'] == value]
        
        # Create a new row with the input column value and means for target columns
        new_row = pd.DataFrame({'Condition': value,
                                'FF(%)': class_df['FF(%)'].values[0],
                                'Eff(%)': class_df['Eff(%)'].values[0],
                               'Jsc(mA/cm^2)': class_df['Jsc(mA/cm^2)'].values[0]}, index=[0])
        
        for column in ['NMP (mL)', 'DMF (mL)', 'DMSO (mL)', 'Perovskite concentration (M)', 
                       'Annealing temperature (℃)', 'Vacuum Pressure (Pa)', 'Vacuum Pressure time (s)', 
                       'Temperature (℃)', 'Humidity (%)', 'Voc(V)']:
            new_row[column] = df.loc[df['Condition'] == value, column].iloc[0]
        
        # Append the new row to the filtered DataFrame
        filtered_df = pd.concat([filtered_df, new_row], ignore_index=True)

    filtered_df = filtered_df[['Condition', 'NMP (mL)', 'DMF (mL)', 'DMSO (mL)', 
                     'Perovskite concentration (M)', 'Annealing temperature (℃)', 
                     'Vacuum Pressure (Pa)', 'Vacuum Pressure time (s)', 
                     'Temperature (℃)', 'Humidity (%)', 'Voc(V)', 'FF(%)', 
                     'Eff(%)', 'Jsc(mA/cm^2)']]

    return filtered_df


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


def define_parameter_space(NMP_options, DMF_options, DMSO_options, PC_options, AT_options,
                                    VP_options, VT_options):
    """ Define the scaled parameter space for the model. """
    return ParameterSpace([DiscreteParameter('NMP', np.arange(0,1.01, 1/(len(NMP_options)-1))),
                                DiscreteParameter('DMF', np.arange(0,1.01, 1/(len(DMF_options)-1))),
                                DiscreteParameter('DMSO', np.arange(0,1.01, 1/(len(DMSO_options)-1))),
                                DiscreteParameter('PC', np.arange(0,1.01, 1/(len(PC_options)-1))),
                                DiscreteParameter('AT', np.arange(0,1.01, 1/(len(AT_options)-1))),
                                DiscreteParameter('VP', np.arange(0,1.01, 1/(len(VP_options)-1))),
                                DiscreteParameter('VT', np.arange(0,1.01, 1/(len(VT_options)-1))),
                                ]) 


def get_matern52_kernel(input_dim):
    """ Return an instance of the matern52 kernel. """
    ker = GPy.kern.Matern52(input_dim = input_dim, ARD = True)
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


def generate_visualization_process_conditions(df, descaled_search_space, name=None, n_panel=4):
    """ Generate a multi-panel figure with bar plots showing the frequency of each process condition. """
    df_cols = df.columns
    n_col = len(df_cols)  # Set n_col to the number of columns in the DataFrame
    num_rows = (n_col + 2) // n_panel  # Calculate the number of rows needed for a 'n_panel'-column layout
    fig, axes = plt.subplots(num_rows, n_panel, figsize=(18, 3.5 * num_rows), sharey=False)
    fs = 24

    for i, col_name in enumerate(df_cols):
        row_idx = i // n_panel
        col_idx = i % n_panel

        ax = axes[row_idx, col_idx]
        ax.hist(df[col_name], bins=20, range=(min(descaled_search_space[i]), max(descaled_search_space[i])), color='dodgerblue', edgecolor='black', alpha=0.9)
        ax.set_xlabel(col_name, fontsize=fs*0.8)
        ax.tick_params(direction='in', length=5, width=1, labelsize=fs * 0.8, grid_alpha=0.5, pad=10)

    # Hide empty subplots
    for i in range(n_col, num_rows * n_panel):
        row_idx = i // n_panel
        col_idx = i % n_panel
        fig.delaxes(axes[row_idx, col_idx])

    for i in range(num_rows):
        axes[i, 0].set_ylabel('count', fontsize=fs*0.8)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    #plt.savefig(f'multi_panel_bar_plots_{name}.png', dpi=600)
    plt.savefig(f'multi_panel_bar_plots_{name}.svg', format='svg')
    plt.show()


def generate_visualization_efficiency_vs_ml_conditions(X_new, Xc, df_device, df_film, f_obj, acq_fcn, 
                                                       acq_cons, acq_produc, descaled_search_space, xlim=45):
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

    X_sorted = x_scaler(df_film.sort_values('Condition').iloc[:,1:8].values, descaled_search_space)
    y_pred, y_uncer = f_obj(X_sorted)
    y_pred = -y_pred[:,-1]
    y_uncer = np.sqrt(y_uncer[:,-1])

    axes[0].scatter(np.arange(1, len(X_sorted) + 1), y_pred,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'gray', label = 'predicted')
    axes[0].errorbar(np.arange(1, len(X_sorted) + 1), y_pred, yerr = y_uncer,  
                ms = 0, ls = '', capsize = 2, alpha = 0.6, color = 'gray', zorder = 0)

    y_pred_new, y_uncer_new = f_obj(X_new)
    y_pred_new = -y_pred_new[:,-1]
    y_uncer_new = np.sqrt(y_uncer_new[:,-1])

    axes[0].scatter(np.arange(len(X_new))+len(Xc) + 1, y_pred_new,
                s = 50, facecolors='none', alpha = 0.6, edgecolor = 'crimson', label = 'suggested')
    axes[0].errorbar(np.arange(len(X_new))+len(Xc) + 1, y_pred_new, yerr = y_uncer_new,  
                 ms = 0, ls = '', capsize = 2, alpha = 0.6, 
                 color = 'crimson', zorder = 0)


    axes[0].set_ylabel('PCE (%)', fontsize = 20)
    axes[0].set_xlabel('Process condition', fontsize = 20)

    axes[0].set_ylim(-1, 30)
    axes[0].set_xlim(-1, xlim)
    axes[0].set_xticks(np.arange(0,xlim,10))
    axes[0].legend(fontsize = fs*0.7)

    axes[1].plot(np.arange(len(X_new))+len(Xc) + 1, acq_cons, marker = 'o',
                ms = 2, alpha = 0.6, color = 'crimson', label = 'constr prob')
    axes[1].plot(np.arange(len(X_new))+len(Xc) + 1, acq_fcn/20, marker = 'o',
                ms = 2, alpha = 0.6, color = 'navy', label = 'raw acqui')

    axes[1].plot(np.arange(len(X_new))+len(Xc) + 1, acq_produc/20, marker = 'o',
                ms = 2, alpha = 0.6, color = 'royalblue', label = 'final acqui')


    axes[1].set_ylim(0.0, 2)
    axes[1].set_xlim(xlim-26, xlim-4)
    axes[1].set_xticks(np.arange(xlim-25,xlim-4,10))
    axes[1].set_ylabel('Acquisition probability', fontsize = fs)
    axes[1].set_xlabel('Process condition', fontsize = fs)

    for ax in axes:
        ax.tick_params(direction='in', length=5, width=1, labelsize = fs*.8, grid_alpha = 0.5)
    plt.subplots_adjust(wspace = 0.4)
    plt.legend(fontsize = fs*0.7)
    plt.savefig(f'efficiency_vs_ml_conditions.svg', format='svg')
    plt.show()


def generate_contour_plot(ind1, ind2, x_sampled, f_obj, x_descaler, x_columns, descaled_search_space):
    """ 
    Generate contour plots showing the evolution of the objective function with respect to two variables

    (starts by sampling 1000 random points, and then modifies the value for variables ind1 and ind2
    -> objective function computed for all of those points and max, mean and min determined 
    -> doing this for a grid of (ind1, ind2)-points enables definition of contour plot for max, mean, min)
    """
    n_steps = 21
    x1x2y_pred = []
    for x1 in np.linspace(0, 1, n_steps):
        for x2 in np.linspace(0, 1, n_steps):
            x_temp = np.copy(x_sampled)
            x_temp[:, ind1] = x1
            x_temp[:, ind2] = x2
            y_pred, _ = f_obj(x_temp)
            y_pred = -y_pred
            x1_org = x_descaler(x_temp, descaled_search_space)[0, ind1]
            x2_org = x_descaler(x_temp, descaled_search_space)[0, ind2]
            x1x2y_pred.append([x1_org, x2_org, np.max(y_pred), np.mean(y_pred), np.min(y_pred)])
            
    x1 = np.array(x1x2y_pred, dtype=object)[:, 0].reshape(n_steps, n_steps)
    x2 = np.array(x1x2y_pred, dtype=object)[:, 1].reshape(n_steps, n_steps)

    y_pred_max = np.array(x1x2y_pred, dtype=object)[:, 2].reshape(n_steps, n_steps)
    y_pred_mean = np.array(x1x2y_pred, dtype=object)[:, 3].reshape(n_steps, n_steps)
    y_pred_min = np.array(x1x2y_pred, dtype=object)[:, 4].reshape(n_steps, n_steps)

    fs = 20
    title_pad = 16

    # Contour for Prediction Efficiency Mean
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=False, sharex=False)
    fig.subplots_adjust(wspace=0.4, hspace=None)
    colorbar_offset = [16, 9, 3]
    for ax, c_offset, y in zip(axes, colorbar_offset,
                               [y_pred_max, y_pred_mean, y_pred_min]):

        c_plt1 = ax.contourf(x1, x2, y, levels=np.arange(19) * 0.25 + c_offset, cmap='coolwarm', extend = 'both')
        cbar = fig.colorbar(c_plt1, ax=ax)
        cbar.ax.tick_params(labelsize=fs * 0.8)
        ax.set_xlabel(str(x_columns[ind1]), fontsize=fs)
        ax.set_ylabel(str(x_columns[ind2]), fontsize=fs)
        ax.tick_params(direction='in', length=5, width=1, labelsize=fs * 0.8)

    axes[0].set_title('Objective fcn max', pad=title_pad, fontsize=fs)
    axes[1].set_title('Objective fcn mean', pad=title_pad, fontsize=fs)
    axes[2].set_title('Objective fcn min', pad=title_pad, fontsize=fs)
    plt.savefig(f'contour_plot_{ind1}_{ind2}.svg', format='svg')
    plt.show()


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
