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

st.title("Concrete Strength Analysis App")

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
        train_indices = range(len(y_train))
        test_indices = range(len(y_train), len(y_train) + len(y_test))
        ax.plot(train_indices, y_train, label=f'Train (R² = {r2_train:.3f})', color='blue')
        ax.plot(train_indices, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
        ax.plot(test_indices, y_test, label=f'Test (R² = {r2_test:.3f})', color='red')
        ax.plot(test_indices, y_test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("Random Forest: Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "random_forest_predictions"), unsafe_allow_html=True)
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
        train_indices = range(len(y_train))
        test_indices = range(len(y_train), len(y_train) + len(y_test))
        ax.plot(train_indices, y_train, label=f'Train (R² = {r2_train:.3f})', color='blue')
        ax.plot(train_indices, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
        ax.plot(test_indices, y_test, label=f'Test (R² = {r2_test:.3f})', color='red')
        ax.plot(test_indices, y_test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("Decision Tree: Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "decision_tree_predictions"), unsafe_allow_html=True)
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
        train_indices = range(len(y_train))
        test_indices = range(len(y_train), len(y_train) + len(y_test))
        ax.plot(train_indices, y_train, label=f'Train (R² = {r2_train:.3f})', color='blue')
        ax.plot(train_indices, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
        ax.plot(test_indices, y_test, label=f'Test (R² = {r2_test:.3f})', color='red')
        ax.plot(test_indices, y_test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("KNN: Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "knn_predictions"), unsafe_allow_html=True)
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
        train_indices = range(len(y_train))
        test_indices = range(len(y_train), len(y_train) + len(y_test))
        ax.plot(train_indices, y_train, label=f'Train (R² = {r2_train:.3f})', color='blue')
        ax.plot(train_indices, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
        ax.plot(test_indices, y_test, label=f'Test (R² = {r2_test:.3f})', color='red')
        ax.plot(test_indices, y_test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("XGBoost: Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "xgboost_predictions"), unsafe_allow_html=True)
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
        train_indices = range(len(y_train))
        test_indices = range(len(y_train), len(y_train) + len(y_test))
        ax.plot(train_indices, y_train, label=f'Train (R² = {r2_train:.3f})', color='blue')
        ax.plot(train_indices, y_train_pred, label='Train Predicted', color='cyan', linestyle='--')
        ax.plot(test_indices, y_test, label=f'Test (R² = {r2_test:.3f})', color='red')
        ax.plot(test_indices, y_test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Concrete Strength (MPa)")
        ax.set_title("AdaBoost: Actual vs Predicted")
        ax.legend(loc='upper left')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "adaboost_predictions"), unsafe_allow_html=True)
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
        st.subheader("SHAP Analysis (Using Random Forest)")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        
        st.write(f"Shape of X_test: {X_test.shape}")
        st.write(f"Shape of shap_values: {shap_values.shape}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test, feature_names=list(X.columns))
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, "shap_summary"), unsafe_allow_html=True)
        plt.close(fig)
        
        fig2 = plt.figure(figsize=(20, 4))
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False, figsize=(20, 4))
        st.pyplot(fig2)
        st.markdown(get_image_download_link(fig2, "shap_force_plot"), unsafe_allow_html=True)
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

else:
    st.write("Please upload an Excel file to begin analysis.")
