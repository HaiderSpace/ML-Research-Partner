import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import shap
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Set page title
st.title("Concrete Strength Analysis App")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (XLSX format)", type=["xlsx"])

# Function to create downloadable PNG link
def get_image_download_link(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename} as PNG</a>'
    return href

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Display the dataframe
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Analysis selection
    analysis_options = [
        "Distribution Graphs",
        "Pairplots",
        "Correlation Heatmap",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "XGBoost",
        "AdaBoost",
        "SHAP Analysis"
    ]
    selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
    
    # Prepare data for ML models
    if selected_analysis in ["Random Forest", "Decision Tree", "KNN", "XGBoost", "AdaBoost", "SHAP Analysis"]:
        X = df.drop('concrete_compressive_strength', axis=1)
        y = df['concrete_compressive_strength']
        # Ensure no NaN values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform selected analysis
    if selected_analysis == "Distribution Graphs":
        st.subheader("Distribution Graphs for All Variables")
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes = axes.ravel()
        for idx, column in enumerate(df.columns):
            sns.histplot(df[column], ax=axes[idx], kde=True)
            axes[idx].set_title(column)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "distribution_graphs"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "Pairplots":
        st.subheader("Pairplots")
        fig = sns.pairplot(df)
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "pairplots"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "correlation_heatmap"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "Random Forest":
        st.subheader("Random Forest Regression")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Mean Squared Error: {mse:.3f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title("Random Forest: Predicted vs Actual")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "random_forest_predictions"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "Decision Tree":
        st.subheader("Decision Tree Regression")
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Mean Squared Error: {mse:.3f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title("Decision Tree: Predicted vs Actual")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "decision_tree_predictions"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "KNN":
        st.subheader("K-Nearest Neighbors Regression")
        knn = KNeighborsRegressor()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Mean Squared Error: {mse:.3f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title("KNN: Predicted vs Actual")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "knn_predictions"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "XGBoost":
        st.subheader("XGBoost Regression")
        xgb = XGBRegressor(random_state=42)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Mean Squared Error: {mse:.3f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title("XGBoost: Predicted vs Actual")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "xgboost_predictions"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "AdaBoost":
        st.subheader("AdaBoost Regression")
        ada = AdaBoostRegressor(random_state=42)
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"R² Score: {r2:.3f}")
        st.write(f"Mean Squared Error: {mse:.3f}")
        
        # Scatter plot of predictions
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title("AdaBoost: Predicted vs Actual")
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "adaboost_predictions"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "SHAP Analysis":
        st.subheader("SHAP Analysis (Using Random Forest)")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        
        # Debugging shapes
        st.write(f"Shape of X_test: {X_test.shape}")
        st.write(f"Shape of shap_values: {shap_values.shape}")
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test, feature_names=list(X.columns))
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "shap_summary"), unsafe_allow_html=True)
        plt.close(fig)
        
        # Force plot for first prediction
        fig2 = plt.figure(figsize=(12, 4))
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "shap_force_plot"), unsafe_allow_html=True)
        plt.close(fig2)

else:
    st.write("Please upload an Excel file to begin analysis.")
