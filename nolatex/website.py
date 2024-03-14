import streamlit as st
import requests

'''
# Taxi Fare Predictions
'''

'''
## Details of the ride
'''
with st.form("my_form"):
    pickup_date = st.date_input("Pick-up Date")
    pickup_time = st.time_input("Pick-up Time")
    pickup_datetime = str(pickup_date) + " " + str(pickup_time)

    pickup_longitude =st.number_input("Pick-up Longitude", value=40.7614327, step=.0000001, format='%.7f')
    pickup_latitude = st.number_input("Pick-up Latitude", value=-73.9798156, step=.0000001, format='%.7f')

    dropoff_longitude =st.number_input("Drop-off Longitude", value=40.6513111, step=.0000001, format='%.7f')
    dropoff_latitude = st.number_input("Drop-off Latitude", value=-73.8803331, step=.0000001, format='%.7f')

    passenger_count = st.slider("Number of Passengers", min_value=1, max_value=8, value=2)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        "---"
        url = 'https://taxifare.lewagon.ai/predict'

        params = {
            'pickup_datetime': pickup_datetime,
            'pickup_longitude': pickup_longitude,
            'pickup_latitude': pickup_latitude,
            'dropoff_longitude': dropoff_longitude,
            'dropoff_latitude': dropoff_latitude,
            'passenger_count': passenger_count
        }

        r = requests.get(url, params=params)
        pred_fare = round(float(dict(r.json())['fare']), 2)

        '''
        ## Your Prediction
        '''
        st.markdown(f"**Fare:** ***{pred_fare}***")
