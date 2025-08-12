import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# Simulated subgroup performance data
data = pd.DataFrame({
    'Group': ['Age 65+ with Diabetes', 'Under 40, No Chronic Conditions', 'Female, Hypertension'],
    'Sensitivity': [0.79, 0.85, 0.88],
    'Specificity': [0.88, 0.92, 0.90],
    'C_Statistic': [0.76, 0.81, 0.83],
    'Lift': [2.9, 3.5, 3.1]
})

# Simulate ROC curve data
def generate_roc_data(group_name, seed=0):
    np.random.seed(seed)
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) + np.random.normal(0, 0.02, 100)
    tpr = np.clip(tpr, 0, 1)
    return pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Group': group_name})

roc_data = pd.concat([
    generate_roc_data('Age 65+ with Diabetes', seed=1),
    generate_roc_data('Under 40, No Chronic Conditions', seed=2),
    generate_roc_data('Female, Hypertension', seed=3)
])

# UI
st.title("Predictive Model Accuracy Dashboard")
st.subheader("Evaluate model performance across patient subgroups")

# Global metrics
st.markdown("### Global Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sensitivity", "0.87")
col2.metric("Specificity", "0.91")
col3.metric("C-Statistic", "0.82")
col4.metric("Lift (Top Decile)", "3.2")

# Sidebar filter
st.sidebar.header("Filter Patient Groups")
selected_group = st.sidebar.selectbox("Select Group", data['Group'])

# Selected group metrics
group_data = data[data['Group'] == selected_group].iloc[0]
st.markdown(f"### Metrics for: {selected_group}")
st.write({
    "Sensitivity": group_data['Sensitivity'],
    "Specificity": group_data['Specificity'],
    "C-Statistic": group_data['C_Statistic'],
    "Lift": group_data['Lift']
})

# Sensitivity chart
st.markdown("### Sensitivity by Group")
chart = alt.Chart(data).mark_bar().encode(
    x='Group',
    y='Sensitivity',
    color='Group'
).properties(width=600)
st.altair_chart(chart, use_container_width=True)

# ROC curve
st.markdown("### ROC Curve")
roc_chart = alt.Chart(roc_data[roc_data['Group'] == selected_group]).mark_line().encode(
    x='FPR',
    y='TPR'
).properties(width=600)
st.altair_chart(roc_chart, use_container_width=True)

# Confusion matrix
st.markdown("### Confusion Matrix")
conf_matrix = np.array([[85, 15], [10, 90]])  # [TN, FP], [FN, TP]
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'], ax=ax)
st.pyplot(fig)
