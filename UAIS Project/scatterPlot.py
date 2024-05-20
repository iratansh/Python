"""
Scatterplot of risk level by sight in factory
"""

import pandas as pd
import matplotlib.pyplot as plt

def get_data_from_OFS():
    """
    Access data from excel file
    Returns data
    """
    file = pd.read_excel('OFS_summary_gyemark.xls', sheet_name="Site Leadership", usecols=['Level', 'Area 1', 'Area 2'])
    return file

def scatterplot_for_sight_risklevel(file):
    """
    Scatterplot for area and risk level
    """
    risk_level = list(file['Level'])
    area = list(file['Area 1'])
    plt.scatter(area, risk_level)
    plt.show()

def scatterplot_for_factors_risk(file):
    """
    Scatterplot for factors during incident and risk
    """
    risk_level = list(file['Level'])
    area = list(file['Area 1'])
    plt.scatter(area, risk_level)
    plt.show()

def scatterplt_for_factors_risk_2(file):
    risk_level = list(file['Level'])
    area = list(file['Area 2'])
    plt.scatter(area, risk_level)
    plt.show()

def main(): 
    """
    Main Function
    """
    data = get_data_from_OFS()
    scatterplot_for_sight_risklevel(data)
    # scatterplt_for_factors_risk_2(data)

main()

    

