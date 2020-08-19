# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import time
import read_data
import build_features
import ensemble_model
import os
import torch
from os import listdir
from os.path import isfile, join

#  Introduction & Overview
st.title('GSA AI/ML Challenge 2020')
st.header('Guidehouse Submission')
st.subheader('Overview')

"""
Welcome to the Guidehouse EULA Evaluation tool. This application allows GSA contracting 
officers to upload .pdf and .docx end-user license agreements (EULAS), which are then parsed 
and analyzed against a trained model to predict acceptable or unacceptable clauses. 
For more information, please review our Technical Documentation and Solution Demonstration.
"""
# File Upload
st.subheader('File Upload')

# Instructions
"""
 Upload one or multiple .docx and .pdf files using the "Select a file" feature below.
 Files must be uploaded to the '/src' folder where the application is currently running.
 The "Clear file list" button resets the files you have selected.
"""
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_static_list():
    list_of_files = [] # this list is initialized once and can be used to store multiple uploaded files
    """This list is initialized once and can be used to store the files uploaded"""
    return list_of_files

def upload():
    """
    Allows user to select one or multiple files for upload
    
    Returns
    -------
    my_list: list
    a list of user-uploaded .docx and/or .pdf files
    """
    my_list = get_static_list() # Create an empty list 
    folder_path = "." # Use src file path
    
    # Get list of only files within the folder path
    onlyfiles = [f for f in listdir(folder_path) if ((isfile(join(folder_path, f)) and f[-4:] == "docx") or (isfile(join(folder_path, f)) and f[-3:] == "pdf"))]
    selected_filename = st.selectbox('Select a .docx or .pdf file', onlyfiles) # Issue with selectbox that shows first file by default, also assumes file is in working directory
    abs_file = os.path.abspath(selected_filename) # Get absolute path for read_EULAs input
    
    if abs_file: 
        if not abs_file in my_list: # Check to see if file is not already in the current list
            my_list += [abs_file] # If not, then add it to the list
    else:
        my_list.clear()  # Clear list if the user clears the cache and reloads the page
        st.info("Upload one or more `.docx` and/or `.pdf` files.")

    if st.button("Clear file list"):
        my_list.clear()
    #if st.checkbox("Show file list?", True):
    #    st.write(list(my_list))
    return my_list

selected_list = upload()
# Use read_EULAs from read_data to read and parse EULAs
uploaded_clauses = read_data.read_EULAs(selected_list)
st.subheader("Uploaded File Preview")

"""
Each uploaded file is parsed into individual clauses for evaluation. The clauses will be used to generate
features, which will be fed into the predictive model. To view the parsed, uploaded files, 
simply select the checkbox below, which will display a table of the clauses in the sidebar.

***Uploaded EULA files:***
"""

# For each selected file, display the content in a table format    
for filename, lst_clauses in uploaded_clauses.items():
    st.write(filename)
    df_clauses = pd.DataFrame(lst_clauses,columns=['Clauses'])
   
# Check the box to show EULAs in a table in the sidebar
if st.checkbox('Check to display parsed EULAs in the sidebar'):
    #st.subheader('EULA files')
    st.sidebar.table(df_clauses)
  
st.subheader('EULA Clause Evaluation')

"""
First, the evaluation tool generates features based on the uploaded EULA clauses.
Once features are generated, they are passed into the trained model to generate a 
clause-by-clause prediction of acceptable or unnacceptable with an accompanied confidence score.\n
*Important note: The feature generation process is resource-intensive, and as a result,
may require more than 2-3 minutes to run. The time required varies depending on the number of
EULA files and features, as well as resource bandwidth. Please refrain from refreshing the application
page while the process is running.*

To initiate the evaluation, click the "Predict!" button below.
"""

feature_button = st.button("Predict!")
def predict():
    if feature_button:
        #bar = st.progress(0)
        # Build features
        with st.spinner("Building features..."):
            features = build_features.gen_fts(uploaded_clauses)
        st.success("Features successfuly created.")
        
        with st.spinner("Modeling..."):
            # Init ensemble model
            model = ensemble_model.EnsembleClass(features)
            # Create prediction
            arr_prediction = model.predict_clause()
            # Create probability
            arr_probability = model.predict_probability()
            # Create dictionary of probability results
            lst_probability = []
            for arr in arr_probability:
                lst_probability.append([prob[1] for prob in arr])
                # combine clauses, prediction, probability into one table
            dict_output = {}
            i = 0
            for key, value in uploaded_clauses.items():
                arr_prediction_int = np.array(arr_prediction[i]).astype(int)
                lst_probability_round = np.array(lst_probability[i]).round(decimals=3)
                pd_output = pd.DataFrame([value,arr_prediction_int,lst_probability_round]).T
                pd_output.columns = ['Text','Prediction','Probability']
                dict_output[key] = pd_output
                i+=1
        st.success("Prediction complete!")
        
        st.subheader("View Results")
        st.markdown("""
                    Below, for each uploaded EULA file, you can view the text for each clause along
                    with its associated prediction and probability. A prediction of '0'
                    means that the model has understood the clause to be acceptable. Conversely,
                    a prediction of '1' indicates that the model has understood it as unacceptable.
                    """)

        for key, value in dict_output.items():
            st.write(key)
            st.table(value)
            
        st.markdown("""
                    If you would like to upload different EULA files, you may return to the
                    beginning of the application and click the "Clear file list" button. 
                    Thank you for using the Guidehouse EULA Evaluation tool!
                    """)
        return dict_output
    
output = predict()    
    
