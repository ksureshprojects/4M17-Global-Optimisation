import numpy as np
import matplotlib.pyplot as plt
from Schubert import *

def plot_min_err(res, title=None, save=None):
    """
    Function to plot minimisation error against evaluations.
    
    Parameters:
    res - Array with objective function evaluations of 
          points visited.
    title - Title string for plots produced
    save - Filename under which to save plots
    """
    it = np.linspace(1,res.shape[1],res.shape[1])
    plt.figure()
    # Plot difference between objective values and 
    # minimum value against evaluations.
    plt.bar(it, res[0] - np.min(res))
    plt.xlabel('Iteration')
    plt.ylabel('Minimisation Error')
    plt.title(title)
    if save is not None:
        plt.savefig(save)
    plt.show()

def tuning_alg(cl, parameters, keywords, title, save, lg=False):
    """
    Function to evaluate effect of changing single parameter
    in a given optimisation algorithm.
    
    Parameters:
    cl - Class corresponding to optimisation method operation
    parameters - Array containing parameter values to test
    keywords - Contains keyword arguments for optimisation 
               run function.
    title - Title string for plots produced
    save - Filename under which to save plots
    """
    # Problem initialisation
    schubert_5d = Schubert(5)
    # Constant starting location
    x0 = np.array([[1],[1],[1],[1],[1]]).astype('float64')
    # Objective function reference
    obj = schubert_5d.eval_obj
    # Evaulate all parameter values over two different
    # random seeds
    for j in range(2):
        # Container to store minimum values
        min_vals = np.array([[]])
        # Container to store iteration number
        iterations = np.array([[]])
        # Run algorithm for each parameter value
        for i,v in enumerate(parameters):
            print('\r {}/{}%'.format((i + 1), parameters.shape[0]), end='')
            # Create dictionary to pass keyword arguments
            # to objective function.
            kw = {keywords[0]:j, keywords[1]:v}
            # Initialise algorithm class
            ins = cl(obj, x0, **kw)
            # Obtain result
            x_opt_a, o_min_a, rest, rest2, its = ins.run()
            # Update minimum values with most recent result
            min_vals = np.block([[min_vals, o_min_a]])
            # Update iterations with most recent result
            iterations = np.block([[iterations, its]])
        
        if lg:
            parameters2 = np.log(parameters)/np.log(10)
            xl = 'log(Parameter Value)'
        else:
            parameters2 = parameters.copy()
            xl = 'Parameter Value'
        # Plot min val results
        plt.figure()
        plt.plot(np.round(parameters2, 2), np.round(min_vals[0],2))
        plt.xlabel(xl)
        plt.ylabel('Minimum Result')
        plt.title(title)
        plt.savefig(save + '_vals_{}'.format(j))
        plt.show()
        # Plot iteration results
        plt.figure()
        plt.plot(np.round(parameters2, 2), iterations[0])
        plt.xlabel(xl)
        plt.ylabel('Number of Obj Evaluations')
        plt.title(title)
        plt.savefig(save + '_its_{}'.format(j))
        plt.show()

def initialisation_effect(cl, title, save):
    """
    Function to evaluate effect of changing initial input
    in a given optimisation algorithm.
    
    Parameters:
    cl - Class corresponding to optimisation method operation
    title - Title string for plots produced
    save - Filename under which to save plots
    """
    # Problem initialisation
    schubert_5d = Schubert(5)
    # Random vector generator
    r_gen = np.random.RandomState(seed=1).uniform
    # Objective function reference
    obj = schubert_5d.eval_obj
    # Evaulate all values over two different
    # random seeds
    for j in range(2):
        # Container to store minimum values
        min_vals = np.array([[]])
        # Container to store iteration number
        iterations = np.array([[]])
        # Run algorithm for each parameter value
        for i in range(40):
            print('\r {}/{}%'.format((i + 1), 40), end='')
            # Generate new initial point
            x0 = r_gen(low=-2, high=2, size=(5,1))
            # Initialise algorithm class
            ins = cl(obj, x0, seed=j)
            # Obtain result
            x_opt_a, o_min_a, rest, rest2, its = ins.run()
            # Update minimum values with most recent result
            min_vals = np.block([[min_vals, o_min_a]])
            # Update iterations with most recent result
            iterations = np.block([[iterations, its]])
        # Plot min val results
        plt.figure()
        plt.plot([i + 1 for i in range(40)], np.round(min_vals[0],2))
        plt.ylabel('Minimum Result')
        plt.title(title)
        plt.savefig(save + '_vals_{}'.format(j))
        plt.show()
        # Plot iteration results
        plt.figure()
        plt.plot([i + 1 for i in range(40)], iterations[0])
        plt.ylabel('Number of Obj Evaluations')
        plt.title(title)
        plt.savefig(save + '_its_{}'.format(j))
        plt.show()