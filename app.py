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
from groq import Groq


st.title("ML-Research Partner")

uploaded_file = st.file_uploader("Upload your Excel file (XLSX format)", type=["xlsx"])

def get_image_download_link(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename} as PNG</a>'
    return href

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    analysis_options = [
        "Distribution Graphs",
        "Pairplots",
        "Correlation Heatmap",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "XGBoost",
        "AdaBoost",
        "SHAP Analysis",
        "Combined Actual vs Predicted",
        "Actual vs Predicted (All Models)"
    ]
    selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
    
    if selected_analysis in ["Random Forest", "Decision Tree", "KNN", "XGBoost", "AdaBoost", "SHAP Analysis", "Combined Actual vs Predicted", "Actual vs Predicted (All Models)"]:
        X = df.drop('concrete_compressive_strength', axis=1)
        y = df['concrete_compressive_strength']
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    
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
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Train Mean Squared Error: {mse_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        st.write(f"Test Mean Squared Error: {mse_test:.3f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_train_pred, label=f'Train (R² = {r2_train:.3f})', color='blue', alpha=0.5)
        ax.scatter(y_test, y_test_pred, label=f'Test (R² = {r2_test:.3f})', color='red', alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect Fit')
        z_train = np.polyfit(y_train, y_train_pred, 1)
        p_train = np.poly1d(z_train)
        ax.plot(y_train, p_train(y_train), color='blue', linestyle='-', label='Train Fit')
        z_test = np.polyfit(y_test, y_test_pred, 1)
        p_test = np.poly1d(z_test)
        ax.plot(y_test, p_test(y_test), color='red', linestyle='-', label='Test Fit')
        ax.set_xlabel("Actual Concrete Strength (MPa)")
        ax.set_ylabel("Predicted Concrete Strength (MPa)")
        ax.set_title("Random Forest: Parity Plot")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "random_forest_parity"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax2.plot(range(len(y_train)), y_train, label='Train Actual', color='green')
        ax2.plot(range(len(y_train)), y_train_pred, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual', color='purple')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Concrete Strength (MPa)")
        ax2.set_title("Random Forest: Actual vs Predicted (Combined)")
        ax2.legend(loc='upper left')
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "random_forest_combined"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "Decision Tree":
        st.subheader("Decision Tree Regression")
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Train Mean Squared Error: {mse_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        st.write(f"Test Mean Squared Error: {mse_test:.3f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_train_pred, label=f'Train (R² = {r2_train:.3f})', color='blue', alpha=0.5)
        ax.scatter(y_test, y_test_pred, label=f'Test (R² = {r2_test:.3f})', color='red', alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect Fit')
        z_train = np.polyfit(y_train, y_train_pred, 1)
        p_train = np.poly1d(z_train)
        ax.plot(y_train, p_train(y_train), color='blue', linestyle='-', label='Train Fit')
        z_test = np.polyfit(y_test, y_test_pred, 1)
        p_test = np.poly1d(z_test)
        ax.plot(y_test, p_test(y_test), color='red', linestyle='-', label='Test Fit')
        ax.set_xlabel("Actual Concrete Strength (MPa)")
        ax.set_ylabel("Predicted Concrete Strength (MPa)")
        ax.set_title("Decision Tree: Parity Plot")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "decision_tree_parity"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax2.plot(range(len(y_train)), y_train, label='Train Actual', color='green')
        ax2.plot(range(len(y_train)), y_train_pred, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual', color='purple')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Concrete Strength (MPa)")
        ax2.set_title("Decision Tree: Actual vs Predicted (Combined)")
        ax2.legend(loc='upper left')
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "decision_tree_combined"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "KNN":
        st.subheader("K-Nearest Neighbors Regression")
        knn = KNeighborsRegressor()
        knn.fit(X_train, y_train)
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Train Mean Squared Error: {mse_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        st.write(f"Test Mean Squared Error: {mse_test:.3f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_train_pred, label=f'Train (R² = {r2_train:.3f})', color='blue', alpha=0.5)
        ax.scatter(y_test, y_test_pred, label=f'Test (R² = {r2_test:.3f})', color='red', alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect Fit')
        z_train = np.polyfit(y_train, y_train_pred, 1)
        p_train = np.poly1d(z_train)
        ax.plot(y_train, p_train(y_train), color='blue', linestyle='-', label='Train Fit')
        z_test = np.polyfit(y_test, y_test_pred, 1)
        p_test = np.poly1d(z_test)
        ax.plot(y_test, p_test(y_test), color='red', linestyle='-', label='Test Fit')
        ax.set_xlabel("Actual Concrete Strength (MPa)")
        ax.set_ylabel("Predicted Concrete Strength (MPa)")
        ax.set_title("KNN: Parity Plot")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "knn_parity"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax2.plot(range(len(y_train)), y_train, label='Train Actual', color='green')
        ax2.plot(range(len(y_train)), y_train_pred, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual', color='purple')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Concrete Strength (MPa)")
        ax2.set_title("KNN: Actual vs Predicted (Combined)")
        ax2.legend(loc='upper left')
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "knn_combined"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "XGBoost":
        st.subheader("XGBoost Regression")
        xgb = XGBRegressor(random_state=42)
        xgb.fit(X_train, y_train)
        y_train_pred = xgb.predict(X_train)
        y_test_pred = xgb.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Train Mean Squared Error: {mse_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        st.write(f"Test Mean Squared Error: {mse_test:.3f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_train_pred, label=f'Train (R² = {r2_train:.3f})', color='blue', alpha=0.5)
        ax.scatter(y_test, y_test_pred, label=f'Test (R² = {r2_test:.3f})', color='red', alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect Fit')
        z_train = np.polyfit(y_train, y_train_pred, 1)
        p_train = np.poly1d(z_train)
        ax.plot(y_train, p_train(y_train), color='blue', linestyle='-', label='Train Fit')
        z_test = np.polyfit(y_test, y_test_pred, 1)
        p_test = np.poly1d(z_test)
        ax.plot(y_test, p_test(y_test), color='red', linestyle='-', label='Test Fit')
        ax.set_xlabel("Actual Concrete Strength (MPa)")
        ax.set_ylabel("Predicted Concrete Strength (MPa)")
        ax.set_title("XGBoost: Parity Plot")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "xgboost_parity"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax2.plot(range(len(y_train)), y_train, label='Train Actual', color='green')
        ax2.plot(range(len(y_train)), y_train_pred, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual', color='purple')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Concrete Strength (MPa)")
        ax2.set_title("XGBoost: Actual vs Predicted (Combined)")
        ax2.legend(loc='upper left')
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "xgboost_combined"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "AdaBoost":
        st.subheader("AdaBoost Regression")
        ada = AdaBoostRegressor(random_state=42)
        ada.fit(X_train, y_train)
        y_train_pred = ada.predict(X_train)
        y_test_pred = ada.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Train Mean Squared Error: {mse_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        st.write(f"Test Mean Squared Error: {mse_test:.3f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_train_pred, label=f'Train (R² = {r2_train:.3f})', color='blue', alpha=0.5)
        ax.scatter(y_test, y_test_pred, label=f'Test (R² = {r2_test:.3f})', color='red', alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect Fit')
        z_train = np.polyfit(y_train, y_train_pred, 1)
        p_train = np.poly1d(z_train)
        ax.plot(y_train, p_train(y_train), color='blue', linestyle='-', label='Train Fit')
        z_test = np.polyfit(y_test, y_test_pred, 1)
        p_test = np.poly1d(z_test)
        ax.plot(y_test, p_test(y_test), color='red', linestyle='-', label='Test Fit')
        ax.set_xlabel("Actual Concrete Strength (MPa)")
        ax.set_ylabel("Predicted Concrete Strength (MPa)")
        ax.set_title("AdaBoost: Parity Plot")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "adaboost_parity"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax2.plot(range(len(y_train)), y_train, label='Train Actual', color='green')
        ax2.plot(range(len(y_train)), y_train_pred, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual', color='purple')
        ax2.plot(range(len(y_train), len(y_train) + len(y_test)), y_test_pred, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Concrete Strength (MPa)")
        ax2.set_title("AdaBoost: Actual vs Predicted (Combined)")
        ax2.legend(loc='upper left')
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "adaboost_combined"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "SHAP Analysis":
        st.subheader("SHAP Analysis (Using XGBoost)")
        xgb = XGBRegressor(random_state=42)
        xgb.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)
        
        st.write(f"Shape of X_test: {X_test.shape}")
        st.write(f"Shape of shap_values: {shap_values.shape}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test, feature_names=list(X.columns))
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "shap_summary"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test, feature_names=list(X.columns), plot_type="bar")
        ax2.set_title("SHAP Mean Value Bar Chart (XGBoost)")
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "shap_mean_bar"), unsafe_allow_html=True)
        plt.close(fig2)
    
    elif selected_analysis == "Combined Actual vs Predicted":
        st.subheader("Combined Actual vs Predicted (Using Random Forest)")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        st.write(f"Train R² Score: {r2_train:.3f}")
        st.write(f"Test R² Score: {r2_test:.3f}")
        
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        y_train_series = pd.Series(y_train_pred)
        y_test_series = pd.Series(y_test_pred)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = range(len(y_actual))
        ax.plot(indices, y_actual, label='Actual', color='green')
        ax.plot(range(len(y_train_pred)), y_train_series, label=f'Train Predicted (R² = {r2_train:.3f})', color='blue', linestyle='--')
        ax.plot(range(len(y_train), len(y_train) + len(y_test_pred)), y_test_series, label=f'Test Predicted (R² = {r2_test:.3f})', color='red', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("Combined Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "combined_actual_vs_predicted"), unsafe_allow_html=True)
        plt.close(fig)
    
    elif selected_analysis == "Actual vs Predicted (All Models)":
        st.subheader("Actual vs Predicted (All Models)")
        models = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42)
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        indices = range(len(y_train) + len(y_test))
        y_actual = pd.concat([pd.Series(y_train), pd.Series(y_test)]).reset_index(drop=True)
        ax.plot(indices, y_actual, label='Actual', color='green')
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            y_pred = pd.concat([pd.Series(y_train_pred), pd.Series(y_test_pred)]).reset_index(drop=True)
            ax.plot(indices, y_pred, label=f'{name} Predicted (Train R² = {r2_train:.3f}, Test R² = {r2_test:.3f})', linestyle='--')
        
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("Actual vs Predicted (All Models)")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "all_models_actual_vs_predicted"), unsafe_allow_html=True)
        plt.close(fig)
    
# Graph Explanation Section for Research-Oriented Output
st.subheader("Graph Explanation (Research-Level Insights)")

graph_options = [
    "Distribution Graphs",
    "Pairplots",
    "Correlation Heatmap",
    "Random Forest Parity Plot",
    "Random Forest Actual vs Predicted",
    "Decision Tree Parity Plot",
    "Decision Tree Actual vs Predicted",
    "KNN Parity Plot",
    "KNN Actual vs Predicted",
    "XGBoost Parity Plot",
    "XGBoost Actual vs Predicted",
    "AdaBoost Parity Plot",
    "AdaBoost Actual vs Predicted",
    "SHAP Summary Plot",
    "SHAP Mean Value Bar Chart",
    "Combined Actual vs Predicted",
    "Actual vs Predicted (All Models)"
]

selected_graph = st.selectbox("Select Graph to Explain", graph_options)

# Securely stored API key (replace with environment variable in production)
groq_api_key = "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"

if st.button("Get Explanation") and selected_graph:
    try:
        client = Groq(api_key=groq_api_key)

        graph_descriptions = {
            "Distribution Graphs": "Distribution plots of input variables (e.g., Cement, Water, Fly Ash, etc.) and the target variable (Compressive Strength) for assessing skewness, kurtosis, and variable spread.",
            "Pairplots": "Pairwise scatter plots between all variables to evaluate linearity, feature interdependency, and potential clusters in the dataset.",
            "Correlation Heatmap": "Correlation matrix visualizing pairwise Pearson correlations among all variables including Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age, and Compressive Strength.",
            "Random Forest Parity Plot": "Actual vs predicted compressive strength comparison using Random Forest. Highlights model bias, variance, and regression accuracy.",
            "Random Forest Actual vs Predicted": "Time-sequenced plot showing Random Forest predictions against actual compressive strength values.",
            "Decision Tree Parity Plot": "Decision Tree model’s parity analysis reflecting overfitting/underfitting behavior.",
            "Decision Tree Actual vs Predicted": "Temporal match between actual and Decision Tree predicted compressive strengths.",
            "KNN Parity Plot": "Parity analysis of K-Nearest Neighbors model showing local generalization capabilities.",
            "KNN Actual vs Predicted": "Prediction alignment of KNN model across sample index.",
            "XGBoost Parity Plot": "XGBoost parity graph showcasing fit quality and performance variation.",
            "XGBoost Actual vs Predicted": "Prediction time-series of XGBoost model over dataset samples.",
            "AdaBoost Parity Plot": "AdaBoost model's performance evaluation via parity analysis.",
            "AdaBoost Actual vs Predicted": "Actual vs predicted values sequence for AdaBoost performance monitoring.",
            "SHAP Summary Plot": "SHAP summary plot for XGBoost revealing individual feature impact and directionality on predictions.",
            "SHAP Mean Value Bar Chart": "Bar chart showing mean absolute SHAP values, reflecting average feature importance in XGBoost predictions.",
            "Combined Actual vs Predicted": "Overlay of actual values and Random Forest predictions for overall visual performance.",
            "Actual vs Predicted (All Models)": "Comprehensive comparison of model predictions vs actual compressive strength across all algorithms."
        }

        description = graph_descriptions.get(selected_graph, "Graph description not available.")

        prompt = (
            f"Assume you are a senior machine learning researcher with 20 years of experience in civil and structural engineering. "
            f"You are writing a scientific interpretation for a peer-reviewed paper on concrete compressive strength prediction. "
            f"Based on the following graph type: '{selected_graph}', provide a highly technical and insightful interpretation. "
            f"Focus only on the *results*, not on the visual elements of the graph. Interpret the graph analytically, "
            f"mentioning specific variables (e.g., Cement, Water, Fly Ash, Age) and their influence. Classify variables "
            f"as having high, moderate, low, or no impact. Identify any model behavior such as overfitting, underfitting, variance, "
            f"generalization gap, or predictive strength. Your explanation must be in formal scientific tone suitable for "
            f"top journals like 'Cement and Concrete Composites', 'Engineering Structures', or 'Construction and Building Materials'. "
            f"Here is the context of the graph: {description}"
        )

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        explanation = chat_completion.choices[0].message.content
        st.subheader(f"Research-Level Interpretation of {selected_graph}")
        st.write(explanation)

    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")


else:
    st.write("Please upload an Excel file to begin analysis.")
