import pandas as pd

# Sample patient subgroup performance data
data = pd.DataFrame({
    'Group': ['Age 65+ with Diabetes', 'Under 40, No Chronic Conditions', 'Female, Hypertension'],
    'Sensitivity': [0.79, 0.85, 0.88],
    'Specificity': [0.88, 0.92, 0.90],
    'C_Statistic': [0.76, 0.81, 0.83],
    'Lift': [2.9, 3.5, 3.1]
})
