import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Load the data
@st.cache_data
def load_data():
    gold_data = pd.read_csv("gld_price_data.csv")
    return gold_data

gold_data = load_data()

# Preprocess the data
gold_data = gold_data.copy()  # Create a copy to ensure immutability
gold_data['Date'] = pd.to_datetime(gold_data['Date'])

# Prepare features and target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X_train, Y_train)

# Prediction function
def predict_gold_price(spx, uso, slv, eur_usd):
    input_data = [[spx, uso, slv, eur_usd]]
    prediction = regressor.predict(input_data)
    return prediction[0]

# App Navigation
st.title("SUV's Gold Price Prediction App")

# Sidebar Navigation
page = st.sidebar.radio("Select a Page", ("Home", "Prediction", "Dataset", "Correlation Heatmap", "Distribution of GLD", "Model Performance"))

# Home Page (Default Page)
if page == "Home":
    st.subheader("Welcome to Gold Price Prediction App!!!")
    st.write("""
        This app allows you to predict gold prices based on various input features such as SPX, USO, SLV, and EUR/USD.
        Use the sidebar to navigate through different pages:
        
        - **Home**: The default landing page with an overview of the app.
        - **Prediction**: Predict gold prices based on user input.
        - **Dataset**: View the dataset used for training the model.
        - **Correlation Heatmap**: Explore the correlation between different features.
        - **Distribution of GLD**: Analyze the distribution of gold prices.
        - **Model Performance**: Evaluate the performance of the model using R-squared score.
    """)
    

# Prediction Page
elif page == "Prediction":
    st.subheader("Gold Price Prediction")
    
    # Input features for prediction on the Prediction page
    st.write("Enter the values for the following features to predict the gold price:")
    col1, col2 = st.columns(2)
    with col1:
        spx = st.number_input("SPX Value", value=gold_data['SPX'].mean())
        uso = st.number_input("USO Value", value=gold_data['USO'].mean())
    with col2:
        slv = st.number_input("SLV Value", value=gold_data['SLV'].mean())
        eur_usd = st.number_input("EUR/USD Value", value=gold_data['EUR/USD'].mean())
    
    if st.button("Predict Gold Price"):
        prediction = predict_gold_price(spx, uso, slv, eur_usd)
        st.success(f"Predicted Gold Price (GLD): {prediction:.2f}")

# Dataset Page
elif page == "Dataset":
    st.subheader("Gold Price Dataset")
    st.write(gold_data.head(15))

# Correlation Heatmap Page
elif page == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    correlation = gold_data.corr(numeric_only=True)
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
    st.pyplot(plt)

# Distribution of GLD Page
elif page == "Distribution of GLD":
    st.subheader("Distribution of GLD")
    plt.figure(figsize=(8, 6))
    sns.histplot(gold_data['GLD'], color='blue', kde=True)
    st.pyplot(plt)

# Model Performance Page
elif page == "Model Performance":
    st.subheader("Model Performance")
    test_data_prediction = regressor.predict(X_test)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    st.write(f"R-squared Error: {error_score:.4f}")
    
