import matplotlib.pyplot as plt
import mpltern.datasets
import numpy as np
import pandas as pd
from baybe import Campaign
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe.searchspace import SearchSpace
from baybe.recommenders import TwoPhaseMetaRecommender, BotorchRecommender, RandomRecommender
'''
Design of Experiments Bayesian Optimisation SAFT Test Environment
Authors: M. Awan, H. A. Fraser, K. Henderson, A. Shearer, R. Shetty, B. Timlin
'''
#Specify API System ---
print ('Available Solvents for API Binary Blend Optimisation: Water, Ethanol, IPA')
print ('Available APIs: Paracetamol, Ibuprofen, Mefanamic Acid')
API = input("Choose API from list of available APIs: ")
if API in ['Paracetamol', 'Ibuprofen', 'Mefanamic Acid']:
    print (f'{API} selected')
else:
    raise ValueError('API not recognised! Check spelling.')
sheetname = API #Define what API sheet should test environment chooses

#Bayesian Optimisation Model Setup ---
#Defining single objective function as solubility
target = NumericalTarget(name="Solubility", mode="MAX")
objective = SingleTargetObjective(target=target)
#Defining categorical and continuous parameters
from baybe.parameters import (CategoricalParameter, NumericalContinuousParameter)
parameters = [
    CategoricalParameter(
        name="Solubility Blend",
        values=["IPA-EtOH", "H2O-EtOH", "IPA-H2O"],
        encoding="OHE", 
    ),
    NumericalContinuousParameter(
    name="Solvent Ratio",
    bounds=(0, 1),
    ),]
#Defining searchspace as combination of all parameters
searchspace = SearchSpace.from_product(parameters=parameters)
# Defining recommenders and acquisition_functions
recommender = TwoPhaseMetaRecommender(
    initial_recommender = RandomRecommender(),
    recommender = BotorchRecommender(acquisition_function="qLogEI"))
#Starting optimisation campaign and creating table for recording iterations
campaign = Campaign(searchspace, objective, recommender)
table = pd.DataFrame()

#Defining function to run a single iteration of the optimisation ---
def iteration():
    '''
    Function to perform each experiemnent and retrieve solubility result from SAFT prediction excel file.
    '''
    global table
    run = campaign.recommend(1) #specifying number of run per iteration
    print("\n\nRecommended conditions for next iteration: ")
    run['Solvent Ratio'] = run['Solvent Ratio'].round(2)
    print (run)
    # Retrieving test environment from SAFT Data
    xlsx = pd.read_excel(r"C:\Users\rohan\OneDrive - University of Leeds\VS Code\VS Code\TPFlash_Results.xlsx", sheet_name = sheetname)
    #Determining column and row positions for SAFT result
    s1, s2 = run.iloc[0, 0].split('-')
    sol_ratio = run.iloc[0,1]
    r_idx = int((sol_ratio*100)+1)
    for column in xlsx.columns:
        if (s1 in column) and (s2 in column):
            c_idx = xlsx.columns.get_loc(column)
            break
    c_idx += 1
    result = xlsx.iloc[r_idx,c_idx]
    print (f'The mole fraction solubility meaasurement is {result}')

    run['Solubility'] = [result]
    campaign.add_measurements(run)
    table = pd.concat([table, run], ignore_index=True)

#Specifying how many iterations to run and looping ---
total_iterations = 24

for x in range(total_iterations):
    iteration()

print (table)

#Converting solvent ratio into volume fractions for logic and ternary diagram---
def convert_table(row):
    '''
    Function to convert the solubility raito into volume fracftions
    '''
    blend = row['Solubility Blend'].split('-')  # Split the blend into solvents
    solvent_ratio = row['Solvent Ratio']
    # Initialize volume fractions
    v_water = 0
    v_ethanol = 0
    v_IPA = 0
    # Assign fractions based on solvent names
    if 'H2O' in blend[0]:
        v_water = 1 - solvent_ratio
    elif 'EtOH' in blend[0]:
        v_ethanol = 1 - solvent_ratio
    elif 'IPA' in blend[0]:
        v_IPA = 1 - solvent_ratio
    
    if 'H2O' in blend[1]:
        v_water = solvent_ratio
    elif 'EtOH' in blend[1]:
        v_ethanol = solvent_ratio
    elif 'IPA' in blend[1]:
        v_IPA = solvent_ratio
    # Return the new columns
    return pd.Series([v_water, v_ethanol, v_IPA, row['Solubility']], 
                     index=['v_water', 'v_ethanol', 'v_IPA', 'solubility'])

# Apply the function to each row
optimisation_table = table.apply(convert_table, axis=1)
print(optimisation_table)

#converting table to 1D arrays to plot for tenary diagram
optimisation_matrix = optimisation_table.to_numpy()
optimisation_matrix[optimisation_matrix == 0] += 0.03 # Added small off set, to stop points being cut off by the border
o_water = optimisation_matrix[:,0]
o_ethanol = optimisation_matrix[:,1]
o_IPA = optimisation_matrix[:,2]
o_solubility = optimisation_matrix[:,3]

#Generating design Space from SAFT Predictions
df = pd.read_excel(r"C:\Users\rohan\OneDrive - University of Leeds\VS Code\VS Code\SAFT Solubility Predictions for Binary Mixtures.xlsx", sheet_name = sheetname)
SAFT_Matrix = df.to_numpy()

#converting SAFTcolumns into lists for ternary plot
p_water = SAFT_Matrix[:,0]
p_ethanol = SAFT_Matrix[:,1]
p_IPA = SAFT_Matrix[:,2]
p_solubility = SAFT_Matrix[:,3]

#Ternary diagram---------------------------------------------------------------------------------------------------
#Plot dummy design space
fig = plt.figure(figsize=(10.8, 4.8))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
ax = plt.subplot(projection="ternary")
pc = ax.scatter(p_water, p_ethanol, p_IPA, c=p_solubility, cmap='inferno', label='Points', marker ='H', s=1000)
#Grid and tickmarks
ax.taxis.set_tick_params(tick1On=True, colors='C0', grid_color='C0')
ax.laxis.set_tick_params(tick1On=True, colors='C1', grid_color='C1')
ax.raxis.set_tick_params(tick1On=True, colors='C2', grid_color='C2')
#Axis Labels
ax.set_tlabel("Water")
ax.set_llabel("Ethanol")
ax.set_rlabel("IPA")
ax.taxis.label.set_color('C0')
ax.laxis.label.set_color('C1')
ax.raxis.label.set_color('C2')
#ColourBar
cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(pc, cax=cax)
colorbar.set_label(f'Solubility of {API}', rotation=270, va='baseline')
#Optimisation Iteration Addition
ax.scatter(o_water, o_ethanol, o_IPA, color='white', label='Points', edgecolor='black', marker ='H', s=200)
max_solubililty = np.max(optimisation_matrix[:, -1])
max_rows = optimisation_matrix[optimisation_matrix[:, -1] == max_solubililty]
max_water = max_rows[:,0]
max_ethanol = max_rows[:,1]
max_IPA = max_rows[:,2]

ax.scatter(max_water, max_ethanol, max_IPA, color='red', label='Points', marker ='*', s=200)
plt.show()