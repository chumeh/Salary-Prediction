import streamlit as st #streamlit to build the front end
import pickle          # to be able to use the generated pickle object
import numpy as np     # To us numpy function

#loading the save pickle file')

model = pickle.load(open(r'/Users/admin/Vs Code Projects/linear_regression_model.pkl', 'rb'))
#setting the title of the streamlit app interface
st.title('Salary Prediction Tool')

#add brief description
st.write('This app helps prodict salary of future workers based on experience and current payment state')

#adding widget for user to enter  years of experience
years_of_experience = st.number_input('Enter Your Years of Experience:', min_value=0.0, value=1.0, step = 1.0)

#action on button click
if st.button('Predict'):
    # prediction using trained model
    experience_input = np.array([[years_of_experience]]) # convert input to 2D
    prediction = model.predict(experience_input)
    
    # display results
    st.success(f'The Predicted salary for {years_of_experience} Years of Experience is: ${prediction[0]:,.2f}')
# display information about the model
st.write('The model was trained using a dataset of salaries and years of experience')   
