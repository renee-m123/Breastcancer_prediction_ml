# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 22:42:36 2025

@author: USER
"""

import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open(r'C:\Users\USER\OneDrive\Desktop\jupyter\project one response\cancer prediction\trained_model5.sav', 'rb'))

def cancer_prediction(input_data):
    
    #converting input data to numpy array because processing is easier than list
    input_data_as_numpy_array= np.array(input_data)
    #reshape the array
    input_data_reshaped= np.array(input_data).reshape(1, -1)
    
    prediction= loaded_model.predict(input_data_reshaped)
    return prediction
    if prediction [0] == 0:
        print("Breast cancer not present")
    else:
        print("Breast cancer present")
    
        
def main():
    #user interface for the web app
    st.title("Breast Cancer diagnosis")
    mean_radius= st.text_input("Mean radius:")
    mean_texture = st.text_input("Mean texture:")
    mean_perimeter =st.text_input("Mean perimeter:")
    mean_area = st.text_input("Mean area:")
    mean_smoothness = st.text_input("Mean smoothness:")
    
    diagnosis =  ''
    if st.button("Check Diagnosis probability"):
         #diagnosis = cancer_prediction([float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area), float(mean_smoothness)])
         try:
           # Convert inputs to float and make a prediction
           prediction = cancer_prediction([
               float(mean_radius),
               float(mean_texture),
               float(mean_perimeter),
               float(mean_area),
               float(mean_smoothness)
           ])
           if prediction[0] == 0:
               diagnosis = "Breast cancer not present"
           else:
               diagnosis = "Breast cancer present"
         except ValueError:
           diagnosis = "Invalid input! Please enter valid numeric values."
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
         