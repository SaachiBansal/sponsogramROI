import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('best_decision_tree_model7.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")
print(type(model))

# Set up the Streamlit interface
st.title('Random Forest Regressor Prediction')

# Input fields for the features
st.header('Enter the campaign data:')

# Assuming the model expects 3 features as an example
duration = st.number_input('Campaign Duration', value=0.0)
budget = st.number_input('Campaign Budget(in rupees)', value=0.0)

st.header('Enter the influencer data:')

# Assuming the model expects 3 features as an example
followers = st.number_input('Number of followers of the influencer', value=0.0)
reach = st.number_input('Reach of the influencer', value=0.0)
eng_rate = st.number_input('Engagement Rate of the influencer', value=0.0)
prev_camp = st.number_input('Clicks in previous campaigns of the influencer', value=0.0)
avg_likes = st.number_input('Average likes on influencer posts', value=0.0)
avg_comm = st.number_input('Average comments on influencer posts', value=0.0)
post = st.number_input('Number of posts of the influencer', value=0.0)
age = st.number_input("Age of the influencer's account", value=0.0)

# Prediction button
if st.button('Predict'):
    # Create a numpy array of the input features
    input_features = np.array([[budget, duration, followers, reach,
                                eng_rate, prev_camp, avg_likes, avg_comm,
                               post, age]])
    
    # Make the prediction
    prediction = model.predict(input_features)

    roi = (prediction-budget)/budget * 100
    
    # Display the prediction
    st.write(f'The predicted Return on Investment is: {roi}')
