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
st.image(image,width='always')



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


    

Features = ['Age_band_of_driver', 'Number_of_casualties' ,'Area_accident_occured', 'Lanes_or_Medians','Time', 'Vehicle_driver_relation',
    'Road_surface_type','Driving_experience', 'Age_band_of_casualty', 'Light_conditions', 'Type_of_collision', 'Number_of_vehicles_involved']


st.markdown("<h1 style='text-align: center;'>Road Traffic Accident Severity App 🚦</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following info:")

        col1, col2 = st.columns(2)

        with col1:
      
            hour = st.slider("Accident Hour: ", 0, 23, value=0, format="%d")
            casualty_age = st.selectbox("Casualty Age: ", options=options_cage)
            casualties = st.slider("No. of Casualties: ", 1, 8, value=0, format="%d")
            collision = st.selectbox("Select Accident Cause: ", options=options_collision)
            vehicles_involved = st.slider("Vehicles involved: ", 1, 7, value=0, format="%d")
            light_con = st.selectbox("Light condition: ", options=options_lcon)
        with col2:
            vd_relation = st.selectbox("Vehicle-Driver relation: ", options=options_vd_relation)
            driver_age = st.selectbox("Driver Age: ", options=options_dage)
            accident_area = st.selectbox("Accident Area: ", options=options_acc_area)
            driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
            lanes = st.selectbox("Lanes: ", options=options_lanes)
            roadsurface_type = st.selectbox("Select Vehicle Type: ", options=options_rsurface_type)
        
        
        submit = st.form_submit_button("Predict the Severity")

    

    if submit:
        light_con = ordinal_encoder(light_con, options_lcon)
        collision = ordinal_encoder(collision, options_collision)
        rsurface_type = ordinal_encoder(roadsurface_type, options_rsurface_type)
        relation = ordinal_encoder(vd_relation, options_vd_relation)
        driver_age =  ordinal_encoder(driver_age, options_dage)
        casualty_age =  ordinal_encoder(casualty_age, options_cage)
        accident_area =  ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        lanes = ordinal_encoder(lanes, options_lanes)


        data = np.array([driver_age,casualties,accident_area,lanes,hour,relation, 
                            rsurface_type,driving_experience,casualty_age,light_con,collision,vehicles_involved]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.markdown("""<style> .big-font { font-family:sans-serif; color:Grey; font-size: 50px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">{pred} is predicted.</p>', unsafe_allow_html=True)
        #st.write(f" => {pred} is predicted. <=")

        p, shap_values = explain_model_prediction(data,dfce)
        st.subheader('Severity Prediction Interpretation Plot')
        st_shap(p)


if __name__ == '__main__':
    main()