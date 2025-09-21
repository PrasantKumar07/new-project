import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, but provide fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP library is not installed. Some explainability features will be limited.")

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #FF4B4B; text-align: center;}
    .section-header {font-size: 2rem; color: #1F77B4; border-bottom: 2px solid #1F77B4;}
    .feature-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .pred-box {padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;}
    .fraud-pred {background-color: #ffcccc; border: 2px solid #FF4B4B;}
    .non-fraud-pred {background-color: #ccffcc; border: 2px solid #00CC96;}
    .metric-box {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 5px;}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # In a real app, you would load your data here
    # For now, we'll create a sample dataset
    # Replace this with your actual data loading code
    try:
        df = pd.read_csv('credit card.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'credit card.csv' is in the same directory.")
        st.stop()

# Preprocess data
def preprocess_data(df):
    # One-hot encode the 'type' column
    df = pd.get_dummies(df, columns=['type'], prefix=['type'])
    
    # Drop identifier columns
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    # Create new features
    df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['transaction_amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    return df

# Train models
def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Isolation Forest': IsolationForest(random_state=42, contamination=0.003),
    }
    
    trained_models = {}
    for name, model in models.items():
        if name in ['Isolation Forest']:
            # For anomaly detection models, we need to fit only on non-fraud data
            non_fraud_idx = y_train == 0
            model.fit(X_train[non_fraud_idx])
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    if model_name in ['Isolation Forest']:
        y_pred = model.predict(X_test)
        # Convert predictions: 1 for normal, -1 for anomaly -> 0 for normal, 1 for fraud
        y_pred = np.where(y_pred == 1, 0, 1)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return y_pred, accuracy, precision, recall, f1

# Generate feature importance explanation (alternative to SHAP)
def explain_prediction_alt(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
        ax.set_title('Top 10 Important Features')
        plt.tight_layout()
        return fig
    elif hasattr(model, 'coef_'):
        # For linear models
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.coef_[0]
        }).sort_values('importance', key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
        ax.set_title('Top 10 Important Features (Absolute Coefficient Values)')
        plt.tight_layout()
        return fig
    return None

# Main app
def main():
    st.markdown('<h1 class="main-header">AI-Powered Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("""
    This advanced fraud detection system uses multiple machine learning algorithms to identify potentially fraudulent transactions.
    Explore the data, train models, and get real-time predictions with explainable AI insights.
    """)
    
    # Load data
    df = load_data()
    processed_df = preprocess_data(df)
    
    # Sidebar
    st.sidebar.header("Navigation")
    app_section = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Model Training", "Real-time Prediction", "AI Explanations"])
    
    # Prepare features and target
    X = processed_df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
    y = processed_df['isFraud']
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if app_section == "Data Overview":
        st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", df.shape[0])
        with col2:
            st.metric("Fraudulent Transactions", df['isFraud'].sum())
        with col3:
            fraud_percentage = (df['isFraud'].sum() / df.shape[0]) * 100
            st.metric("Fraud Percentage", f"{fraud_percentage:.4f}%")
        with col4:
            st.metric("Features", processed_df.shape[1])
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Data Description")
        st.dataframe(df.describe())
        
        # Data imbalance visualization
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        fraud_counts = df['isFraud'].value_counts()
        ax[0].bar(['Non-Fraud', 'Fraud'], fraud_counts.values, color=['green', 'red'])
        ax[0].set_title('Fraud vs Non-Fraud Transactions')
        ax[0].set_ylabel('Count')
        
        # Pie chart
        ax[1].pie(fraud_counts.values, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%', colors=['green', 'red'])
        ax[1].set_title('Fraud Proportion')
        
        st.pyplot(fig)
    
    elif app_section == "Data Analysis":
        st.markdown('<h2 class="section-header">Data Analysis</h2>', unsafe_allow_html=True)
        
        # Transaction types
        st.subheader("Transaction Types Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df['type'].value_counts(), 
                         title='Transaction Types Distribution',
                         labels={'value': 'Count', 'index': 'Transaction Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fraud_by_type = df.groupby('type')['isFraud'].mean().reset_index()
            fig = px.bar(fraud_by_type, x='type', y='isFraud',
                         title='Fraud Rate by Transaction Type',
                         labels={'isFraud': 'Fraud Rate', 'type': 'Transaction Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Amount analysis
        st.subheader("Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter out extreme values for better visualization
            amount_filtered = df[df['amount'] < df['amount'].quantile(0.99)]
            fig = px.histogram(amount_filtered, x='amount', color='isFraud',
                               title='Transaction Amount Distribution by Fraud Status',
                               nbins=50, barmode='overlay', opacity=0.7,
                               labels={'amount': 'Amount', 'isFraud': 'Is Fraud'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fraud_amounts = df[df['isFraud'] == 1]['amount']
            non_fraud_amounts = df[df['isFraud'] == 0]['amount']
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=non_fraud_amounts, name='Non-Fraud', marker_color='green'))
            fig.add_trace(go.Box(y=fraud_amounts, name='Fraud', marker_color='red'))
            fig.update_layout(title='Transaction Amount by Fraud Status')
            st.plotly_chart(fig, use_container_width=True)
        
        # PCA visualization
        st.subheader("PCA Visualization of Transactions")
        
        # Scale the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_scaled)
        
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['isFraud'] = y_train.reset_index(drop=True)
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='isFraud',
                         title='PCA of Transaction Data',
                         labels={'isFraud': 'Is Fraud'},
                         color_continuous_scale=['green', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    elif app_section == "Model Training":
        st.markdown('<h2 class="section-header">Model Training & Evaluation</h2>', unsafe_allow_html=True)
        
        st.info("""
        Multiple machine learning models are trained to detect fraudulent transactions. 
        The models include both supervised and anomaly detection algorithms.
        """)
        
        # Model selection
        model_options = ['Random Forest', 'Logistic Regression', 'Isolation Forest']
        selected_models = st.multiselect('Select models to train and compare', model_options, default=['Random Forest', 'Logistic Regression'])
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Train models
                trained_models = train_models(X_train_scaled, y_train)
                
                # Evaluate models
                results = []
                for name in selected_models:
                    y_pred, accuracy, precision, recall, f1 = evaluate_model(
                        trained_models[name], X_test_scaled, y_test, name
                    )
                    results.append({
                        'Model': name,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("Model Performance Comparison")
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
                
                # Visualization of metrics
                metrics_df = results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                fig = px.bar(metrics_df, x='Model', y='Score', color='Metric', barmode='group',
                             title='Model Performance Metrics')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show confusion matrix for the best model (by F1 score)
                best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
                best_model = trained_models[best_model_name]
                y_pred, _, _, _, _ = evaluate_model(best_model, X_test_scaled, y_test, best_model_name)
                
                st.subheader(f"Confusion Matrix - {best_model_name}")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # ROC Curve (for supervised models only)
                if best_model_name in ['Random Forest', 'Logistic Regression']:
                    st.subheader(f"ROC Curve - {best_model_name}")
                    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                            name=f'ROC curve (AUC = {roc_auc:.2f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                            name='Random', line=dict(dash='dash')))
                    fig.update_layout(
                        title='Receiver Operating Characteristic (ROC) Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        width=700, height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save models for later use
                for name, model in trained_models.items():
                    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                
                st.success("Models trained and saved successfully!")
    
    elif app_section == "Real-time Prediction":
        st.markdown('<h2 class="section-header">Real-time Fraud Prediction</h2>', unsafe_allow_html=True)
        
        # Load trained models if available
        try:
            rf_model = joblib.load('random_forest_model.pkl')
            lr_model = joblib.load('logistic_regression_model.pkl')
            scaler = joblib.load('scaler.pkl')
            models_loaded = True
        except:
            st.warning("Please train models first in the 'Model Training' section.")
            models_loaded = False
        
        if models_loaded:
            st.subheader("Enter Transaction Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                step = st.number_input("Step (hour)", min_value=1, max_value=744, value=1)
                amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=0.01)
                oldbalance_org = st.number_input("Old Balance Origin", min_value=0.0, value=1000.0, step=0.01)
                newbalance_orig = st.number_input("New Balance Origin", min_value=0.0, value=900.0, step=0.01)
            
            with col2:
                oldbalance_dest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0, step=0.01)
                newbalance_dest = st.number_input("New Balance Destination", min_value=0.0, value=0.0, step=0.01)
                is_flagged_fraud = st.selectbox("Is Flagged Fraud", [0, 1])
                transaction_type = st.selectbox("Transaction Type", df['type'].unique())
            
            with col3:
                # Create derived features
                balance_change_orig = oldbalance_org - newbalance_orig
                balance_change_dest = newbalance_dest - oldbalance_dest
                transaction_amount_ratio = amount / (oldbalance_org + 1e-9)  # Avoid division by zero
                
                st.metric("Balance Change Origin", f"{balance_change_orig:.2f}")
                st.metric("Balance Change Destination", f"{balance_change_dest:.2f}")
                st.metric("Amount to Balance Ratio", f"{transaction_amount_ratio:.4f}")
            
            # Create input data for prediction
            input_data = pd.DataFrame({
                'step': [step],
                'amount': [amount],
                'oldbalanceOrg': [oldbalance_org],
                'newbalanceOrig': [newbalance_orig],
                'oldbalanceDest': [oldbalance_dest],
                'newbalanceDest': [newbalance_dest],
                'isFlaggedFraud': [is_flagged_fraud],
                'balance_change_orig': [balance_change_orig],
                'balance_change_dest': [balance_change_dest],
                'transaction_amount_ratio': [transaction_amount_ratio]
            })
            
            # One-hot encode the transaction type
            for t in df['type'].unique():
                input_data[f'type_{t}'] = [1 if transaction_type == t else 0]
            
            # Ensure all columns are present
            for col in X.columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match training data
            input_data = input_data[X.columns]
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            if st.button("Predict Fraud"):
                # Make predictions with both models
                rf_pred = rf_model.predict(input_data_scaled)
                rf_pred_proba = rf_model.predict_proba(input_data_scaled)
                
                lr_pred = lr_model.predict(input_data_scaled)
                lr_pred_proba = lr_model.predict_proba(input_data_scaled)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Random Forest Prediction")
                    if rf_pred[0] == 1:
                        st.markdown('<div class="pred-box fraud-pred">⚠️ Fraudulent transaction detected!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pred-box non-fraud-pred">✅ Legitimate transaction</div>', unsafe_allow_html=True)
                    st.metric("Fraud Probability", f"{rf_pred_proba[0][1]:.2%}")
                
                with col2:
                    st.subheader("Logistic Regression Prediction")
                    if lr_pred[0] == 1:
                        st.markdown('<div class="pred-box fraud-pred">⚠️ Fraudulent transaction detected!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pred-box non-fraud-pred">✅ Legitimate transaction</div>', unsafe_allow_html=True)
                    st.metric("Fraud Probability", f"{lr_pred_proba[0][1]:.2%}")
                
                # Show agreement
                if rf_pred[0] == lr_pred[0]:
                    st.success("✅ Models agree on the prediction")
                else:
                    st.warning("⚠️ Models disagree on the prediction")
    
    elif app_section == "AI Explanations":
        st.markdown('<h2 class="section-header">AI Explanation & Interpretability</h2>', unsafe_allow_html=True)
        
        st.info("""
        Understand how the AI models make predictions using feature importance.
        This helps build trust in the model and provides insights into what features drive fraud predictions.
        """)
        
        # Load trained model if available
        try:
            rf_model = joblib.load('random_forest_model.pkl')
            scaler = joblib.load('scaler.pkl')
            model_loaded = True
        except:
            st.warning("Please train the Random Forest model first in the 'Model Training' section.")
            model_loaded = False
        
        if model_loaded:
            st.subheader("Global Feature Importance")
            
            # Calculate and display feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                         title='Top 10 Important Features for Fraud Detection',
                         labels={'importance': 'Importance', 'feature': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Local Explanation for a Transaction")
            
            # Select a transaction from test set
            sample_idx = st.slider("Select a transaction from test set", 0, len(X_test)-1, 0)
            sample_data = X_test.iloc[sample_idx:sample_idx+1]
            sample_data_scaled = scaler.transform(sample_data)
            
            actual_label = y_test.iloc[sample_idx]
            prediction = rf_model.predict(sample_data_scaled)[0]
            prediction_proba = rf_model.predict_proba(sample_data_scaled)[0][1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Actual Label", "Fraud" if actual_label == 1 else "Non-Fraud")
            with col2:
                st.metric("Prediction", "Fraud" if prediction == 1 else "Non-Fraud")
            with col3:
                st.metric("Fraud Probability", f"{prediction_proba:.2%}")
            
            # Show feature values for the selected transaction
            st.subheader("Feature Values for Selected Transaction")
            feature_values = pd.DataFrame({
                'Feature': feature_names,
                'Value': sample_data.values[0]
            })
            st.dataframe(feature_values)
            
            # Generate explanation using alternative method
            if st.button("Generate Explanation"):
                with st.spinner("Generating explanation..."):
                    fig = explain_prediction_alt(rf_model, feature_names)
                    if fig:
                        st.pyplot(fig)
                        
                        st.subheader("Interpretation")
                        st.markdown("""
                        The feature importance plot shows which features contribute most to the model's predictions.
                        
                        - **Higher values**: Features that have more influence on the prediction
                        - The model uses these features to determine if a transaction is fraudulent
                        """)
                    else:
                        st.warning("Feature importance explanation is not available for this model type.")
            
            # Show SHAP message if not available
            if not SHAP_AVAILABLE:
                st.warning("""
                **SHAP is not installed**. For more detailed explanations with SHAP values, please install it using:
                ```
                pip install shap
                ```
                Then restart the application.
                """)

if __name__ == "__main__":
    main()