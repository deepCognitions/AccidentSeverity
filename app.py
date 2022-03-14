from heapq import heappop
from shutil import move
import streamlit as st
import shap
import pickle
import pandas as pd
import numpy as np
import joblib
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from predict import get_prediction, ordinal_encoder
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('Img/rta_img.jpg')



model = joblib.load(r'Model/RF_RTA02.pkl')
#with open('Model/RF_RTA02.pkl', 'rb') as handle:
#    dfce = pickle.load(handle)
#shap.initjs()
dfce = shap.TreeExplainer(model)
      

def explain_model_prediction(data,dfce):
        # Calculate Shap values
        shap_values = dfce.shap_values(data)
        p = shap.force_plot(dfce.expected_value[1], shap_values[1], data)
        return p, shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.set_page_config(page_title="Deep's Road Traffic Accident Severity Prediction",
                   page_icon="🚦", layout="wide")
st.image(image,use_column_width='always')



#creating option list for dropdown menu

options_lcon=['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']
options_collision=['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
    'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles',
    'Collision with pedestrians', 'With Train']
options_rsurface_type=['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress', 'Gravel roads', 'Other']
options_vd_relation=['Employee', 'Owner', 'Other']
options_dage=['18-30', '31-50', 'Over 51', 'Under 18']
options_cage=['31-50', '18-30', 'Between 5&18', 'Over 51', 'Below 5']
options_acc_area=['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Rural village areasOffice areas',
       'Recreational areas']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way', 'Two-way (divided with solid lines road marking)']


    

Features = ['Time','Age_band_of_driver', 'Vehicle_driver_relation','Driving_experience','Area_accident_occured', 'Lanes_or_Medians',
'Road_surface_type', 'Light_conditions',  'Type_of_collision','Number_of_vehicles_involved', 'Number_of_casualties' , 'Age_band_of_casualty']


st.markdown("<h1 style='text-align: center;'>Road Traffic Accident Severity App 🚦</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following info:")

        col1, col2 = st.columns(2)

        with col1:
      
            inp = {}
            
            inp["Time"] = st.slider("Accident Hour: ", 0, 23, value=0, format="%d")
            inp["Age_band_of_driver"] = st.selectbox("Driver Age: ", options=options_dage)
            inp["Vehicle_driver_relation"] = st.selectbox("Vehicle-Driver relation: ", options=options_vd_relation)
            inp["Driving_experience"] = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
            inp["Area_accident_occured"] = st.selectbox("Accident Area: ", options=options_acc_area)
            inp["Lanes_or_Medians"] = st.selectbox("Lanes: ", options=options_lanes)
        with col2:
            inp["Road_surface_type"] = st.selectbox("Road Surface Type: ", options=options_rsurface_type)
            inp["Light_conditions"] = st.selectbox("Light conditions: ", options=options_lcon)
            inp["Type_of_collision"] = st.selectbox("Type of Collision: ", options=options_collision)
            inp["Number_of_vehicles_involved"] = st.slider("Vehicles involved: ", 1, 7, value=0, format="%d")
            inp["Number_of_casualties"] = st.slider("No. of Casualties: ", 1, 8, value=0, format="%d")
            inp["Age_band_of_casualty"] = st.selectbox("Casualty Age: ", options=options_cage)
        
        
        submit = st.form_submit_button("Predict the Severity")

    

    if submit:
        inp["Light_conditions"] = ordinal_encoder(inp["Light_conditions"], options_lcon)
        inp["Type_of_collision"] = ordinal_encoder(inp["Type_of_collision"], options_collision)
        inp["Road_surface_type"] = ordinal_encoder(inp["Road_surface_type"], options_rsurface_type)
        inp["Vehicle_driver_relation"] = ordinal_encoder(inp["Vehicle_driver_relation"], options_vd_relation)
        inp["Age_band_of_driver"] =  ordinal_encoder(inp["Age_band_of_driver"], options_dage)
        inp["Age_band_of_casualty"] =  ordinal_encoder(inp["Age_band_of_casualty"], options_cage)
        inp["Area_accident_occured"] =  ordinal_encoder(inp["Area_accident_occured"], options_acc_area)
        inp["Driving_experience"] = ordinal_encoder(inp["Driving_experience"], options_driver_exp) 
        inp["Lanes_or_Medians"] = ordinal_encoder(inp["Lanes_or_Medians"], options_lanes)

        df = pd.DataFrame.from_dict([inp])
        pred = get_prediction(data=df, model=model)

        st.markdown("""<style> .big-font { font-family:sans-serif; color:Grey; font-size: 50px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">{pred} is predicted.</p>', unsafe_allow_html=True)
        #st.write(f" => {pred} is predicted. <=")

        p, shap_values = explain_model_prediction(df,dfce)
        st.subheader('Severity Prediction Interpretation Plot')
        st_shap(p)


if __name__ == '__main__':
    main()