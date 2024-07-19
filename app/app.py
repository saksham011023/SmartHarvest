import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# Load models and define constants
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                   'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

def weather_fetch(city_name):
    api_key = "your_api_key_here"  # Add your weather API key here
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# Streamlit App
st.title("SmartHarvest")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Crop Recommendation", "Fertilizer Suggestion", "Disease Detection"])

# Home Page
if page == "Home":
    st.header("Welcome to SmartHarvest!")
    st.write("Get informed decisions about your farming strategy.")

# Crop Recommendation Page
if page == "Crop Recommendation":
    st.header("Crop Recommendation")
    N = st.number_input("Nitrogen", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorous", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium", min_value=0, max_value=100, value=50)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, value=100.0)
    city = st.text_input("City Name")

    if st.button("Predict Crop"):
        if weather_fetch(city) is not None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            st.success(f"The recommended crop is: {final_prediction}")
        else:
            st.error("City not found. Please enter a valid city name.")

# Fertilizer Suggestion Page
if page == "Fertilizer Suggestion":
    st.header("Fertilizer Suggestion")
    crop_name = st.selectbox("Crop you want to grow", ["Select crop", "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya", "coconut", "cotton", "jute", "coffee"])
    N = st.number_input("Nitrogen", min_value=0, max_value=100, value=50)
    P = st.number_input("Phosphorous", min_value=0, max_value=100, value=50)
    K = st.number_input("Potassium", min_value=0, max_value=100, value=50)
    
    if st.button("Recommend Fertilizer"):
        df = pd.read_csv('Data/fertilizer.csv')
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]
        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"
        recommendation = fertilizer_dic[key]
        st.success(f"Fertilizer recommendation: {recommendation}")

# Disease Detection Page
if page == "Disease Detection":
    st.header("Disease Detection")
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type="jpg")

    if uploaded_file is not None:
        img = uploaded_file.read()
        prediction = predict_image(img)
        st.image(Image.open(io.BytesIO(img)), caption='Uploaded Image.', use_column_width=True)
        st.write(f"Prediction: {disease_dic[prediction]}")

if __name__ == '__main__':
    main()
