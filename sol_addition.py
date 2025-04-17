import pandas as pd
import numpy as np 

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

def ia_dummy(mass_added, vol_added, solubility, rsd): 
    '''
    Function to dummy responses from image analysis software

    Args:
        mass_added (float): mass added (in mg) initially
        vol_added (float): volume added (in uL) so far
        solubility (float): gProms predicted solubility (mg/uL) of API in solvent blend
        rsd (float): user defined relative standard deviation of the i.a. software
    Returns:
        completely_dissolved (boolean): T/F - "Is the system fully dissolved?"
        dummy (float): dummy value of % dissolved from image analysis software
    '''
    # Calculate how much mass the volume added so far can dissolve
    mass_dissolved = vol_added*solubility
    # Assuming perfect determination of clear point
    # If the mass the volume added can dissolve exceeds the initial mass, stop the run 
    if mass_dissolved >= mass_added:
        # 100% of mass is dissolved
        return True, 1
    # Otherwise, calculate how close we are to the clear point (using 'true' values)
    percent_dissolution = mass_dissolved/mass_added
    # Assume measurement error from image analysis is normally distributed with a user-defined relative standard deviation
    # Repeat until dummy is < 1
    dummy = 1.01
    while dummy > 1:
        dummy = np.random.normal(percent_dissolution, percent_dissolution*rsd)
    return False, dummy

a, b = ia_dummy(500, 400, 1, 0.25)
print(a,b)
