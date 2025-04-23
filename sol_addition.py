import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import linear_model as sklm 

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

def test_algs(algs, API = 'Ibuprofen', s1 = 'EtOH', s2 = 'IPA', sol_ratio = 0.5, initial_mass = 500, rsd = 0.1, init_steps = 5, n_reps = 3, alg_kwargs = [{}]):
    '''
    Function to test the algorithms in the sol_addition.py file
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
        alg_kwargs (list): list of dictionaries of keyword arguments to pass to the algorithms
    Returns:
        results (list): list of lists of results for each algorithm at each iteration
    '''
    # Get solubility from gProms data
    solubility = read_gProms_data(API, s1, s2, sol_ratio)
    # Create a list to store the results for each algorithm
    res = []
    # Create a list to store the performance of each algorithm
    perf = []
    # Loop through each algorithm and run it
    for i in range(len(algs)):
        # Create a list to store the results for this algorithm
        alg_res = []
        # Repeat the test for the number of repetitions specified
        for _ in range(n_reps):
            # Set the initial volume added to 0
            vol_added = 0
            # Set the completely_dissolved flag to False
            completely_dissolved = False
            # Set up array for X and Y values, with (0, 0) as the first point
            run_res = [[0], [0]]
            for step in range(init_steps):
                # Add the minimum volume of solvent (10 uL) to the system
                # Note there is an error of +/- 2 uL in the volume added
                vol_added += 10 + np.random.normal(0, 2)
                # Check dissolution status using the dummy function
                completely_dissolved, percent_dissolution = ia_dummy(initial_mass, vol_added, solubility, rsd)
                # Append the results to the algorithm results list
                run_res[0].append(vol_added), run_res[1].append(percent_dissolution)
            # Now run the algorithm
            # Loop until the system is fully dissolved
            while not completely_dissolved:
                # Get volume to add from the algorithm
                vol_to_add = algs[i](run_res[0], run_res[1], kwargs = alg_kwargs[i])
                # Add the volume to the system
                # Note there is an error of +/- 2 uL in the volume added
                vol_added += vol_to_add + np.random.normal(0, 2)
                # Check dissolution status using the dummy function
                completely_dissolved, percent_dissolution = ia_dummy(initial_mass, vol_added, solubility, rsd)
                # Append the results to the algorithm results list
                run_res[0].append(vol_added), run_res[1].append(percent_dissolution)
            # Append the results for this run to the algorithm results list
            alg_res.append(run_res)
        # Append the results for this algorithm to the overall results list
        res.append(alg_res)
        # Calculate the performance of the algorithm
        # Calculate the mean number of steps taken to reach 100% dissolution
        mean_num_steps = np.mean([len(run[0]) for run in alg_res])
        # Calculate the mean measured solubility
        mean_measured_solubility = np.mean([initial_mass/run[0][-1] for run in alg_res])
        # Calculate the mean error
        mean_error = (mean_measured_solubility - solubility)/solubility 
        # Calculate the mean volume overshoot
        mean_overshoot = np.mean([run[0][-1] - initial_mass/solubility for run in alg_res])
        perf.append({
            'algorithm': algs[i].__name__,
            'mean_num_steps_after_init': f"{mean_num_steps - init_steps:.3f}",
            'mean_measured_solubility': f"{mean_measured_solubility:.3f} mg/uL",
            'real_solubility': f"{solubility:.3f} mg/uL",
            'mean_error': f"{mean_error:.3%}",
            'mean_volume_overshoot': f"{mean_overshoot:.3f} uL",
        })
    return res, perf

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
    # Get the fixed step size from the kwargs
    step_size = kwargs.get('step_size', 0.1)
    # Calculate the volume to add using OLS regression
    reg = sklm.LinearRegression()
    # Fit the regression model to the data
    reg.fit(np.array(X).reshape(-1, 1), Y)
    # Use regression model to predict volume to add to incriment the % dissolution by the step size
    vol_to_add = (step_size)/reg.coef_[0] # We know intercept is 0, so we can ignore it
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
    # Get the fixed step size from the kwargs
    step_size = kwargs.get('step_size', 0.1)
    # Calculate the volume to add using Theil-Sen regression
    reg = sklm.TheilSenRegressor()
    # Fit the regression model to the data
    reg.fit(np.array(X).reshape(-1, 1), Y)
    # Use regression model to predict volume to add to incriment the % dissolution by the step size
    vol_to_add = (step_size)/reg.coef_[0] # We know intercept is 0, so we can ignore it
    return vol_to_add

# test the OLS_fixed_steps algorithm
if __name__ == '__main__':
    # Set up the parameters for the test
    API = 'Ibuprofen'
    s1 = 'EtOH'
    s2 = 'IPA'
    sol_ratio = 0.5
    initial_mass = 500
    rsd = 0.1
    init_steps = 5
    n_reps = 3
    algs = [OLS_fixed_steps, TheilSen_fixed_steps] 
    alg_kwargs = [[{'step_size': 0.1}], [{'step_size': 0.1}]] # List of dictionaries of keyword arguments for each algorithm
    # Run the test
    res, perf = test_algs(algs=algs, API=API, s1=s1, s2=s2, sol_ratio=sol_ratio, initial_mass=initial_mass, rsd=rsd, init_steps=init_steps, n_reps=n_reps, alg_kwargs=alg_kwargs)
    # Print the results and performance
    print(perf)