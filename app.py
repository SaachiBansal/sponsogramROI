import streamlit as st
import pickle
import numpy as np
import sys
print(f"Python executable: {sys.executable}")
import sys
print(f"Python executable: {sys.executable}")

# Load the saved model
with open('lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")
print(type(model))

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded successfully!")

# Set up the Streamlit interface
st.title('ROI Prediction')


st.title('ROI Prediction')



st.header('Enter the campaign data:')

duration = st.number_input('Campaign Duration', value=0)
duration = st.number_input('Campaign Duration', value=0)
budget = st.number_input('Campaign Budget(in rupees)', value=0.0)

st.header('Enter the influencer data:')

followers = st.number_input('Number of followers of the influencer', value=0.0)
eng_rate = st.number_input('Engagement Rate of the influencer', value=0.0)
prev_camp = st.number_input('Clicks in previous campaigns of the influencer', value=0.0)
avg_likes = st.number_input('Average likes on influencer posts', value=0.0)
avg_comm = st.number_input('Average comments on influencer posts', value=0.0)

totalMetrics = followers*duration*budget/100000

totalMetrics = followers*duration*budget/100000

# Prediction button
if st.button('Predict'):
    # Create a numpy array of the input features
    input_features = np.array([[followers,prev_camp, 
                                eng_rate, avg_likes, avg_comm, totalMetrics]])
    input_features = np.array([[followers,prev_camp, 
                                eng_rate, avg_likes, avg_comm, totalMetrics]])
    
    # Make the prediction
    prediction = model.predict(input_features)

    roi = (prediction-budget)/budget
    transformed_roi = np.array([roi])

    transformed_roi = scaler.transform(transformed_roi)
        
        # Display the prediction
    # st.write(f'prediction {prediction}')
    # st.write(f'roi {roi}')
    # st.write(f'Transformed roi {transformed_roi}')
    if(roi < 0):
        st.write(f'The predicted Return on Investment is poor')
    elif(roi>=0 and roi < 0.3):
        st.write(f'The predicted Return on Investment is below average')
    elif(roi>=0.3 and roi <0.7):
        st.write(f'The predicted Return on Investment is average')
    elif(roi>=0.7 and roi <1.1):
        st.write(f'The predicted Return on Investment is good')
    elif(roi>=1.1 and roi <1.7):
        st.write(f'The predicted Return on Investment is very good')
    else:
        st.write(f'The predicted Return on Investment is excellent')
    
        
