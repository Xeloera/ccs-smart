import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model as sklm 
from sklearn import preprocessing as skpp
from sklearn import gaussian_process as skgp
from itertools import combinations
import time

'''
Solvent addition algorithm test environment for CCS Living Lab Project
Authors: M. Awan, H. A. Fraser, K. Henderson, A. Shearer, R. Shetty, B. Timlin
'''

def read_gProms_data(API, s1, s2, sol_ratio):
    ''' 
    Function to read TPFlash_Results.xlsx excel sheet and return the solubility in mg/mL for a given solvent ratio 

    Args:
        API (str): API selected for the run (valid values: 'Ibuprofen', 'Mefenamic Acid', 'Paracetamol')
        s1 (str) : solvent 1 selected for the run
        s2 (str) : solvent 2 selected for the run (valid values: 'H2O', 'EtOH', 'IPA')
        sol_ratio (float): solvent ratio selected for the run (this is the ratio of x_s1/(x_s1+x_s2))
    Returns:
        solubility (float): solubility in mg/uL from gProms predictions
    '''
    # Required parameters (MW in g/mol, density @ 25C in g/mL)
    MW = {
        'H2O': 18.01528,
        'EtOH': 46.068,
        'IPA': 60.096,
        'Ibuprofen': 206.29,
        'Mefenamic Acid': 241.28,
        'Paracetamol': 151.163
    }
    density = {        
        'H2O': 1,
        'EtOH': 0.7849,
        'IPA': 0.7812
    }
    # Get correct sheetname
    if (API == 'Ibuprofen') or (API == 'Mefenamic Acid'):
        sheetname = API
    elif API == 'Paracetamol':
        sheetname = 'Paracetamol (New Params.)'
    else: 
        raise ValueError('API not recognised! Check spelling.')
    # Read relevant excel sheet
    xlsx = pd.read_excel('TPFlash_Results.xlsx', sheet_name = sheetname)
    # Pull correct index for the solvent system
    for column in xlsx.columns:
        if (s1 in column) and (s2 in column):
            idx = xlsx.columns.get_loc(column)
            break
    if idx == None: 
        raise ValueError('Solvents not recognised! Check spelling.')
    # The user may input s1/s2 in the opposite order as has been calculated from gProms, so accomodate
    # Check what s1 is on the xlsx sheet
    s1_xlsx = column.split(',')[1].split('= ')[1]
    # Round sol_ratio to 2dp to match formatting from gProms
    sol_ratio = np.round(sol_ratio, 2)
    if s1 == s1_xlsx:
        # Read mole fractions
        x = [xlsx.iat[int(sol_ratio*100 + 1),idx+1], xlsx.iat[int(sol_ratio*100 + 1), idx+2], xlsx.iat[int(sol_ratio*100 + 1), idx+3]]
    else: # If it is the opposite way around, invert the solvent ratio
        sol_ratio = 1 - sol_ratio
        # Read mole fractions
        x = [xlsx.iat[int(sol_ratio*100 + 1),idx+1], xlsx.iat[int(sol_ratio*100 + 1), idx+3], xlsx.iat[int(sol_ratio*100 + 1), idx+2]]
    # Calculating mg/mL solubility using a 1 mol basis:
    # Find mass in grams
    mass_g = [x[0] * MW[API], x[1] * MW[s1], x[2] * MW[s2]]
    # Assuming a linear combination of pure solvent volumes is valid, and ignoring API's contribution to volume
    # This is almost certainly wrong but it doesn't need to be great for this dummy
    vol_mL = (mass_g[1] / density[s1]) + (mass_g[2] / density[s2])
    # Find mg/uL solubility
    solubility_mg_uL = mass_g[0]/vol_mL

    return solubility_mg_uL

def ia_dummy(initial_mass, vol_added, solubility, rsd): 
    '''
    Function to dummy responses from image analysis software

    Args:
        initial_mass (float): mass added (in mg) initially
        vol_added (float): volume added (in uL) so far
        solubility (float): gProms predicted solubility (mg/uL) of API in solvent blend
        rsd (float): user defined relative standard deviation of the i.a. software
    Returns:
        completely_dissolved (boolean): T/F - "Is the system fully dissolved?"
        measured_percent_dissolution (float): dummy value of measured % dissolved from image analysis software
    '''
    # Calculate how much mass the volume added so far can dissolve
    mass_dissolved = vol_added*solubility
    # Assuming perfect determination of clear point
    # If the mass the volume added can dissolve exceeds the initial mass, stop the run 
    if mass_dissolved >= initial_mass:
        # 100% of mass is dissolved
        return True, 1
    # Otherwise, calculate how close we are to the clear point (using 'true' values)
    percent_dissolution = mass_dissolved/initial_mass
    # Assume measurement error from image analysis is normally distributed with a user-defined relative standard deviation
    # Repeat until measured_percent_dissolution is < 1
    measured_percent_dissolution = 1.01
    while measured_percent_dissolution > 1:
        measured_percent_dissolution = np.random.normal(percent_dissolution, percent_dissolution*rsd)
    return False, measured_percent_dissolution

def run_algs_n_reps(algs, API = 'Ibuprofen', s1 = 'EtOH', s2 = 'IPA', sol_ratio = 0.5, initial_mass = 150, rsd = 0.1, init_steps = 5, n_reps = 5, min_volume = 10, alg_kwargs = [{}]):
    '''
    Function to check the algorithms' peformance in the sol_addition.py file on a particular API and solvent system
    This function runs the algorithms for a number of repetitions and returns the mean results for each algorithm
    Args:
        algs (list): list of algorithms to test
        API (str): API selected for the run (valid values: 'Ibuprofen', 'Mefenamic Acid', 'Paracetamol')
        s1 (str) : solvent 1 selected for the run
        s2 (str) : solvent 2 selected for the run (valid values: 'H2O', 'EtOH', 'IPA')
        sol_ratio (float): solvent ratio selected for the run (this is the ratio of x_s1/(x_s1+x_s2))
        initial_mass (float): mass added (in mg) initially
        rsd (float): user defined relative standard deviation of the i.a. software
        init_steps (str): number of initial (minimum volume) steps to take before switching to the algorithm
        n_reps (int): number of repetitions to run the test for each algorithm
        min_volume (int): minimum volume to add at each step (in uL)
        alg_kwargs (list): list of dictionaries of keyword arguments to pass to the algorithms
    Returns:
        results (list): list of lists of results for each algorithm at each iteration
        perf_list (list) : list of dicts containing performance metrics for each algorithm
    '''
    # Get solubility from gProms data
    solubility = read_gProms_data(API, s1, s2, sol_ratio)
    # Create a list to store the results for each algorithm
    res = []
    # Create a dict to store the performance of each algorithm
    perf = {'API': API, 's1': s1, 's2': s2, 'sol_ratio': sol_ratio}
    # Create a list to store the performance metrics for each algorithm
    perf_list = []
    # Loop through each algorithm and run it
    for i in range(len(algs)):
        # Create a list to store the results for this algorithm
        alg_res = []
        # Repeat the test for the number of repetitions specified
        for _ in range(n_reps):
            # Set the initial volume added (both measured and actual) to 0
            measured_vol_added = 0 
            actual_vol_added = 0
            # Set the completely_dissolved flag to False
            completely_dissolved = False
            # Set up array for X and Y values, with (0, 0) as the first point
            run_res = [[0], [0]]
            for step in range(init_steps):
                # Add the minimum volume of solvent (10 uL) to the system
                # Note there is an error of +/- 2 uL in the volume added
                # Track what has been measured to be added (fed to algorithm) and what has actually been added (fed to image analysis dummy) separately 
                measured_vol_added += min_volume
                actual_vol_added += min_volume + np.random.normal(0,2)
                # Check dissolution status using the dummy function
                completely_diss, percent_diss = ia_dummy(initial_mass, actual_vol_added, solubility, rsd)
                # Append the results to the algorithm results list
                # Assume actual amount added is not known, so take it as 10uL (ignoring error)
                run_res[0].append(measured_vol_added), run_res[1].append(percent_diss)
            # Now run the algorithm
            # Loop until the system is fully dissolved
            while not completely_diss:
                # Get volume to add from the algorithm
                vol_to_add = algs[i](run_res[0], run_res[1], **alg_kwargs[i])
                # Add the volume to the system
                # Note there is an error of +/- 2 uL in the volume added
                # Track what has been measured to be added (fed to algorithm) and what has actually been added (fed to image analysis dummy) separately
                measured_vol_added += vol_to_add
                actual_vol_added += vol_to_add + np.random.normal(0,2)
                # Check dissolution status using the dummy function
                completely_diss, percent_diss = ia_dummy(initial_mass, actual_vol_added, solubility, rsd)
                # Append the results to the algorithm results list
                run_res[0].append(measured_vol_added), run_res[1].append(percent_diss)
            # Append the results for this run to the algorithm results list
            alg_res.append(run_res)
        # Append the results for this algorithm to the overall results list
        res.append(alg_res)
        # Calculate the performance of the algorithm
        # Calculate the mean measured solubility
        mean_measured_solubility = np.mean([initial_mass/run[0][-1] for run in alg_res])
        # Calculate the mean error
        mean_error = (mean_measured_solubility - solubility)/solubility 
        # Calculate the mean volume overshoot
        mean_overshoot = np.mean([run[0][-1] - initial_mass/solubility for run in alg_res])
        perf.update({
            'algorithm': algs[i].__name__,
            'mean_num_steps': np.mean([len(run[0]) for run in alg_res]),
            'mean_measured_solubility': mean_measured_solubility,
            'real_solubility': solubility,
            'mean_error': mean_error,
            'mean_volume_overshoot': mean_overshoot,
        })
        perf_list.append(perf.copy())
    return res, perf_list

def OLS_fixed_steps(X, Y, **kwargs):
    '''
    Function to implement the OLS (ordinary least squares) algorithm with fixed steps
    Args:
        X (list): list of X values (volumes added (in uL) at each % dissolution)
        Y (list): list of Y values (percent dissolutions at each volume added)
        **kwargs (dict): dictionary of keyword arguments for the algorithm
              step_size (int): desired % dissolution step size (default = 10%)
    Returns:
        vol_to_add (float): volume to add to the system (in uL)
    '''
    # Get the fixed step size from the kwargs (default to 10%)
    # This is the step size in % dissolution, not volume added
    step_size = kwargs.get('step_size', 0.1)
    # Calculate the volume to add using OLS regression
    reg = sklm.LinearRegression(fit_intercept=False)
    # Fit the regression model to the data
    reg.fit(np.array(X).reshape(-1, 1), Y)
    # Estimate volume required to clear point
    clr_pnt = 1/reg.coef_[0]
    # Estimate current % dissolution
    current_diss = reg.coef_[0] * X[-1]
    # If a step would take the system over the clear point, add volume to reach the clear point
    if current_diss + step_size > 1:
        vol_to_add = clr_pnt - X[-1]
    # Otherwise, add volume to reach the next step size
    else:
        vol_to_add = (step_size) / reg.coef_[0]
    return vol_to_add

def TheilSen_fixed_steps(X, Y, **kwargs):
    '''
    Function to implement the Theil-Sen algorithm with fixed steps
    Args:
        X (list): list of X values (volumes added (in uL) at each % dissolution)
        Y (list): list of Y values (percent dissolutions at each volume added)
        **kwargs (dict): dictionary of keyword arguments for the algorithm
              step_size (int): desired % dissolution step size (default = 10%)
    Returns:
        vol_to_add (float): volume to add to the system (in uL)
    '''
    # Get the fixed step size from the kwargs, default to 10%
    # This is the step size in % dissolution, not volume added
    step_size = kwargs.get('step_size', 0.1)
    # Calculate the volume to add using Theil-Sen regression
    reg = sklm.TheilSenRegressor(fit_intercept=False)
    # Fit the regression model to the data
    reg.fit(np.array(X).reshape(-1, 1), Y)
    # Estimate volume required to clear point
    clr_pnt = 1/reg.coef_[0]
    # Estimate current % dissolution
    current_diss = reg.coef_[0] * X[-1]
    # If a step would take the system over the clear point, add volume to reach the clear point
    if current_diss + step_size > 1:
        vol_to_add = clr_pnt - X[-1]
    # Otherwise, add volume to reach the next step size
    else:
        vol_to_add = (step_size) / reg.coef_[0]
    return vol_to_add

def BR_fixed_steps(X, Y, **kwargs):
    '''
    Function to implement the Bayesian Ridge regression algorithm with fixed steps
    Args:
        X (list): list of X values (volumes added (in uL) at each % dissolution)
        Y (list): list of Y values (percent dissolutions at each volume added)
        **kwargs (dict): dictionary of keyword arguments for the algorithm
              step_size (int): desired % dissolution step size (default = 10%)
    Returns:
        vol_to_add (float): volume to add to the system (in uL)
    '''
    # Get the fixed step size from the kwargs, default to 10%
    # This is the step size in % dissolution, not volume added
    step_size = kwargs.get('step_size', 0.1)
    # Calculate the volume to add using Bayesian Ridge regression
    reg = sklm.BayesianRidge(fit_intercept=False)
    # Fit the regression model to the data
    reg.fit(np.array(X).reshape(-1, 1), Y)
    # Estimate volume required to clear point
    clr_pnt = 1/reg.coef_[0]
    # Estimate current % dissolution
    current_diss = reg.coef_[0] * X[-1]
    # If a step would take the system over the clear point, add volume to reach the clear point
    if current_diss + step_size > 1:
        vol_to_add = clr_pnt - X[-1]
    # Otherwise, add volume to reach the next step size
    else:
        vol_to_add = (step_size) / reg.coef_[0]
    return vol_to_add

# Test the algorithms on pure systems, as well as for a random sample of solvent ratios for each API and solvent system combination
def test_algorithms(algs, alg_kwargs = [{}]): 
    '''
    Function to test the algorithms in the sol_addition.py file
    Args:
        algs (list): list of algorithms to test
    Returns:
        perf_df (df): dataframe of performance metrics for each algorithm
    '''
    APIs = ['Ibuprofen', 'Mefenamic Acid', 'Paracetamol']
    solvents = ['H2O', 'EtOH', 'IPA']
    perf_df = pd.DataFrame()
    # Test the algorithms on pure systems first (do 5 reps for each algorithm)
    for API in APIs:
        for solvent in solvents:
            # Run the algorithms for 5 repetitions
            if solvent != 'IPA':
                res, perf_list = run_algs_n_reps(algs, API, s1 = solvent, s2 = 'IPA', sol_ratio = 1, initial_mass = 1000, rsd = 0.1, init_steps = 5, n_reps = 10, alg_kwargs = alg_kwargs)
            else:
                res, perf_list = run_algs_n_reps(algs, API, s1 = 'IPA', s2 = 'EtOH', sol_ratio = 1, initial_mass = 1000, rsd = 0.1, init_steps = 5, n_reps = 10, alg_kwargs = alg_kwargs)
            # Append the results to the performance dataframe
            for i in range(len(algs)):
                # Get the performance metrics for this algorithm
                perf = perf_list[i]
                # Append the performance metrics to the dataframe
                perf_df = pd.concat((perf_df, pd.DataFrame(perf, index = [0])))
    # Now test the algorithms on a sample of solvent ratios for each API and solvent system combination
    for API in APIs:
        for solvent_pair in combinations(solvents, 2):
            # Generate a uniform sample of 10 solvent ratios between 0 and 1
            sample = np.random.uniform(0.01, 0.99, size=(10, 1)) # Have already ran pures, so exclude 0 and 1
            # Loop through each sample and run the algorithms
            for i in range(len(sample)):
                # Get the solvent ratio from the sample
                sol_ratio = sample[i][0]
                # Run the algorithms for 5 repetitions
                res, perf_list = run_algs_n_reps(algs, API, s1=solvent_pair[0], s2=solvent_pair[1], sol_ratio=sol_ratio, initial_mass=150, rsd=0.1, init_steps=5, n_reps = 10, alg_kwargs = alg_kwargs)
            # Append the results to the performance dataframe
                for i in range(len(algs)):
                    # Get the performance metrics for this algorithm
                    perf = perf_list[i]
                    # Append the performance metrics to the dataframe
                    perf_df = pd.concat((perf_df, pd.DataFrame(perf, index = [0])))
    return perf_df

df = test_algorithms([OLS_fixed_steps, TheilSen_fixed_steps, BR_fixed_steps], alg_kwargs=[{'step_size': 0.1}, {'step_size': 0.1}, {'step_size': 0.1}])
df.to_excel(f'alg_performance_{time.strftime("%Y%m%d_%H%M%S")}.xlsx', index = False)