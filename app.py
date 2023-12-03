import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np

#loading the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

#loading the model
model = load_model('saved_model.h5')

def get_blood_pressure(input_text):
    # Assuming the input is in the format "diastolic / systolic"
    parts = input_text.split('/')
    if len(parts) == 2:
        try:
            diastolic = int(parts[1].strip())
            systolic = int(parts[0].strip())
            return diastolic, systolic
        except ValueError:
            return None, None
    else:
        return None, None

# title for the model 
st.title("Sleep Quality Prediction Web App")

st.header("Please Fill in The Form")

# ['BMI Category_encoded']
Stress_Level = st.slider("Stress Level (1 to 10)", min_value=1, max_value=10, step=1, value=1)

Sleep_Duration = st.number_input("Sleep Duration per day(in hrs)", min_value= 0.0, max_value=12.0, value=0.0)

Age = st.number_input('Age (in years)',min_value=5, max_value=150, step=1, value=10)

Heart_Rate = st.number_input('Average Heart Rate per minute', min_value=50, max_value=150, step=1, value=50)

Physical_Activity_Level = st.slider('Physical Activity Level (1 to 100)', min_value = 0, max_value=100, step = 1, value=0)

Daily_Steps = st.number_input('Average Daily Steps', min_value=10, max_value=70000, value=10)

Blood_Pressure =st.text_input("Enter blood pressure (systolic/diastolic):")
if Blood_Pressure:
    Diastolic, Systolic = get_blood_pressure(Blood_Pressure)
    if Diastolic is  None and Systolic is  None:
        st.error("Invalid input format. Please use systolic/diastolic.")

BMI_Category_encoded_mapping = {0: 'Below Normal weight', 3: 'Overweight', 1: 'Normal Weight',2: 'Obese'}
BMI_Category_encoded = st.selectbox("BMI Category", list(BMI_Category_encoded_mapping.keys()), format_func=lambda x: BMI_Category_encoded_mapping[x])

# creating the dataframe
if st.button('Predict Sleep Quality'):
    user_input = pd.DataFrame({
        'Stress Level': [Stress_Level],
        'Sleep Duration': [Sleep_Duration],
        'Age': [Age],
        'Heart Rate': [Heart_Rate],
        'Physical Activity Level': [Physical_Activity_Level],
        'Daily Steps': [Daily_Steps],
        'Diastolic': [Diastolic],
        'Systolic': [Systolic],
        'BMI Category_encoded': [BMI_Category_encoded]
    })

    # scaling data
    StandardScaler = loaded_scaler
    needed_features = ['Stress Level', 'Sleep Duration', 'Age', 'Heart Rate', 'Physical Activity Level', 'Daily Steps', 'Diastolic', 'Systolic', 'BMI Category_encoded']
    user_input_scaled = StandardScaler.transform(user_input[needed_features])
    user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=needed_features)

    # making prediction using the AI model
    prediction = model.predict(user_input_scaled_df)

    st.subheader("Sleep Quality Prediction")
    sleep_quality_value = np.argmax(prediction)
    st.write(f"Out of 5 your Sleep Quality is rated {sleep_quality_value}")

    st.subheader("Personalised Things To Do To Better your Sleep")
    if sleep_quality_value >= 4:
        st.write("Bravo! Your sleep quality is at an optimal level.")
    else:
        st.write("You should do more to increase your sleep quality")

    if Sleep_Duration >= 8:
        st.write("You are getting the optimal sleep for humans and this is beneficial to your sleep quality")
    else:
        st.write(f"The optimal sleep for humans is 8hours you need to add atleast {8 - Sleep_Duration} hours to your sleep schedule to get the best sleep quality possible")

    if Physical_Activity_Level >= 60:
        st.write("Keep up your activity level this is healthy and good for quality sleep")
    else:
        st.write(f"To get optimal sleep quality, you should increase your activity level from {Physical_Activity_Level} to atleast 60")

    if Stress_Level >= 7:
        st.write("Hey there, take a breather. It is okay to take a break once in awhile. This would improve your sleep quality alot")
    else:
        st.write("You seem to be managing your stress well. This is very beneficial to your sleep.")
    
    if BMI_Category_encoded == 3 or BMI_Category_encoded == 2:
        st.write("Your BMI category is needs a bit of work, especially to increase your sleep quality")
    
    if sleep_quality_value <= 1:
        st.write(f"Your sleep quality is {sleep_quality_value}. This is very low. Try and see a doctor near you for consultation and possible treatment")
    

       
    



