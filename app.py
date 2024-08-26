from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import requests
import os
import google.generativeai as genai
import pickle
import numpy as np
from scrape import script
from chain import conversational_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from googletrans import Translator
from werkzeug.utils import secure_filename
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from model1 import CNN_NeuralNet
import markdown

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key'  # Ensure this is a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'

# In-memory storage for users and cities
users = []
city = []

# Home route
@app.route('/')
def index():
    return render_template('login.html')

# User registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if request.is_json:
            user = request.json
        else:
            user = request.form.to_dict()
        session['language'] = user.get('language', 'en')
        users.append(user)
        return jsonify({"message": "User registered successfully"}), 201
    return render_template('registration.html')

# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        mobile = data.get('mobile')
        password = data.get('password')

        user = next((user for user in users if user['mobile'] == mobile and user['password'] == password), None)

        if user:
            session['language'] = user.get('language', 'en')
            return jsonify({"message": "Login successful!"}), 200
        else:
            return jsonify({"message": "Invalid mobile number or password"}), 401
    return render_template('login.html')

# Route to set language
@app.route('/set_language/<lang>', methods=['GET'])
def set_language(lang):
    session['language'] = lang
    flash('Language updated successfully!')
    return redirect(url_for('welcome'))

# Language logic
def get_language_suffix():
    language = session.get('language', 'en')
    if language == 'en':
        return ''
    elif language == 'mr':
        return '1'
    elif language == 'hi':
        return '2'
    return ''

# Welcome page
messageList = []
@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Save form data (example: you can save it to a database)
        data = {
            'name': name,
            'email': email,
            'message': message
        }
        
        # Flash a success message
        flash('Form submitted successfully!')
        
        return render_template(f'welcome{get_language_suffix()}.html')
    
    user = users[-1] if users else {'name': ''}
    return render_template(f'welcome{get_language_suffix()}.html', name=user['name'])

# Weather page
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        data = request.get_json()
        location = data.get('location')
        weather_data = get_weather_data(location)
        if weather_data:
            return jsonify(weather_data)
        else:
            return jsonify({'error': 'Could not retrieve weather data.'}), 400
    return render_template(f'weather{get_language_suffix()}.html')

# Function to get weather data
def get_weather_data(location):
    api_key = '0931fd55055aa0f7904e511e33994db2'
    if location.isdigit():
        url = f"http://api.openweathermap.org/data/2.5/weather?zip={location},in&appid={api_key}&units=metric"
    else:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    print(response.json())  # Debugging statement
    
    if response.status_code == 200:
        data = response.json()
        city.append(data['name'])
        weather = {
            'location': data['name'],
            'temperature': data['main']['temp'],
            'condition': data['weather'][0]['main'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'icon': data['weather'][0]['icon'],
        }

        # Fetch 5-day forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code == 200:
            forecast_data = forecast_response.json()
            forecast = []
            
            # Collect data for 5 days (every 8th entry for 3-hourly forecast)
            for i in range(0, 40, 8):
                day = forecast_data['list'][i]
                forecast.append({
                    'date': day['dt_txt'],
                    'temp': day['main']['temp'],
                    'icon': day['weather'][0]['icon'],
                    'condition': day['weather'][0]['main']
                })

            return {'weather': weather, 'forecast': forecast}
        else:
            return {'weather': weather, 'forecast': []}
    else:
        return None

# Function to get weather data by coordinates
@app.route('/current-weather', methods=['POST'])
def current_weather():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    weather_data = get_weather_by_coordinates(lat, lon)
    if weather_data:
        return jsonify(weather_data)
    else:
        return jsonify({'error': 'Could not retrieve weather data.'}), 400

def get_weather_by_coordinates(lat, lon):
    api_key = '0931fd55055aa0f7904e511e33994db2'
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    print(response.json())  # Debugging statement
    
    if response.status_code == 200:
        data = response.json()
        city.append(data['name'])
        weather = {
            'location': data['name'],
            'temperature': data['main']['temp'],
            'condition': data['weather'][0]['main'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'icon': data['weather'][0]['icon'],
        }

        # Fetch 5-day forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code == 200:
            forecast_data = forecast_response.json()
            forecast = []
            
            # Collect data for 5 days (every 8th entry for 3-hourly forecast)
            for i in range(0, 40, 8):
                day = forecast_data['list'][i]
                forecast.append({
                    'date': day['dt_txt'],
                    'temp': day['main']['temp'],
                    'icon': day['weather'][0]['icon'],
                    'condition': day['weather'][0]['main']
                })

            return {'weather': weather, 'forecast': forecast}
        else:
            return {'weather': weather, 'forecast': []}
    else:
        return None

# Chatbot page
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_page():
    if request.method == 'POST':
        data = request.json
        user_message = data.get('message')
        bot_reply = get_bot_response(user_message)
        formatted_reply = markdown.markdown(bot_reply)  # Convert markdown to HTML
        return jsonify(reply=formatted_reply)
    return render_template(f'chatbot{get_language_suffix()}.html')

def get_bot_response(user_message):
    language = session.get('language', 'en')
    translator = Translator()

    detected = translator.detect(user_message)
    detected_language = detected.lang

    llm1 = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=None,
        timeout=None, 
        google_api_key='AIzaSyCQbNkygleMD3b6QI1QFq8-Zr9gpMBAfP4'
    )

    try:
        if detected_language == 'en':
            model_out_en = conversational_rag_chain.invoke({"input": user_message}, config={"configurable": {"session_id": "abc123"}})
            bot_reply = model_out_en['answer']
            bot_reply = translator.translate(bot_reply, dest=language).text

        else:
            model_en = translator.translate(user_message, dest='en').text
            model_out_en = conversational_rag_chain.invoke({"input": model_en}, config={"configurable": {"session_id": "abc123"}})
            bot_reply = model_out_en['answer']
            bot_reply = translator.translate(bot_reply, dest=language).text

    except Exception as e:
        bot_reply = str(e)
    
    return bot_reply

@app.route('/commodity')
def commodity_page():
    return render_template('commodity.html')

# Handle commodity requests
@app.route('/request', methods=['GET'])
def requestPage():
    commodity_query = request.args.get('commodity')
    state_query = request.args.get('state')
    market_query = request.args.get('district')
    
    if not commodity_query or not state_query or not market_query:
        return render_template('commodity.html', data={"error": "Missing query parameters"})

    try:
        json_data = script(state_query, commodity_query, market_query)
        return render_template('commodity.html', data=json_data)
    except Exception as e:
        return render_template('commodity.html', data={"error": str(e)})

#---------------------------------------------------------------------------------------------------------------------------------
# Map page
@app.route('/map')
def map_page():
    if city:
        location = city[-1]
        if location.lower() == 'aurangabad':
            location = 'ch. Sambhaji Nagar'
        lat, lng = get_coordinates(location)
        return render_template(f'map{get_language_suffix()}.html', lat=lat, lng=lng)
    else:
        lat, lng = get_coordinates('India')
        return render_template(f'map{get_language_suffix()}.html', lat=lat, lng=lng)

# Function to get coordinates from location
def get_coordinates(location):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key=c2d7dfab21804c06a3b4174bcce2fde6'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['lat']
            lng = data['results'][0]['geometry']['lng']
            return lat, lng
    return None, None

#----------------------------------------------------------------------------------------------------------------------------
# Load the crop recommendation model
with open('crop_recommedation.pkl', 'rb') as f:
    model = pickle.load(f)

# List of crops
crop_list = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee"
]

# Crop prediction route
@app.route('/crop-predict', methods=['GET', 'POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['potassium'])  # Corrected 'potassium' spelling
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city = request.form.get("city")

            weather = weather_fetch(city)
            if weather is not None:
                temperature, humidity = weather
                values = [N, P, K, temperature, humidity, ph, rainfall]
                print(values)
                data = np.array([values])
                prediction = model.predict(data)
                final_prediction = crop_list[prediction[0]]
                print(final_prediction)
                language = session.get('language', 'en')
                translator = Translator()
                final_prediction= translator.translate(final_prediction, dest=language).text
                return render_template(f'crop{get_language_suffix()}.html', prediction=final_prediction)
            else:
                error = "Weather data not available for the specified city."
                return render_template(f'crop{get_language_suffix()}.html', error=error)

        except ValueError:
          #  error = "Invalid input. Please enter numeric values for N, P, K, pH, and rainfall."
            return render_template(f'crop{get_language_suffix()}.html')
        
      # Clear session on initial load or refresh
    return render_template(f'crop{get_language_suffix()}.html')


# Function to fetch weather data for crop prediction
def weather_fetch(city_name):
    api_key = "0931fd55055aa0f7904e511e33994db2"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "q=" + city_name + "&appid=" + api_key
    response = requests.get(complete_url)

    if response.status_code == 200:
        data = response.json()
        main = data['main']
        temperature = main['temp'] - 273.15  # Convert from Kelvin to Celsius
        humidity = main['humidity']
        return temperature, humidity
    else:
        return None
#----------------------------------------------------------------------------------------------------------
# PDF listing and display
@app.route('/pdfs')
def show_pdfs():
    # List of states and corresponding PDF filenames
    pdf_files = {
        'Maharashtra': 'maharashtra.pdf',
        'Haryana': 'haryana.pdf',
        'Gujarat': 'gujarat.pdf',
        'Andhra Pradesh': 'andhrapradesh.pdf',
        'Karnataka': 'karnataka.pdf',
        'Kerala': 'kerala.pdf',
        'Tamil Nadu': 'tamilnadu.pdf'
    }
    return render_template(f'pdf{get_language_suffix()}.html', pdf_files=pdf_files)

@app.route('/view-pdf', methods=['GET'])
def view_pdf():
    pdf_file = request.args.get('pdf')
    return render_template(f'view_pdf.html{get_language_suffix()}', pdf_file=pdf_file) 
    
#------------------------------------------------------------------------------------------------------------------------------
translator = Translator()

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple:healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_names = {
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Cedar Apple Rust',
    'Apple:healthy': 'Healthy Apple',
    'Blueberry___healthy': 'Healthy Blueberry',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Healthy Cherry',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Healthy Corn',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight (Isariopsis Leaf Spot)',
    'Grape___healthy': 'Healthy Grape',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange Huanglongbing (Citrus Greening)',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Healthy Peach',
    'Pepper,_bell___Bacterial_spot': 'Bell Pepper Bacterial Spot',
    'Pepper,_bell___healthy': 'Healthy Bell Pepper',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Healthy Potato',
    'Raspberry___healthy': 'Healthy Raspberry',
    'Soybean___healthy': 'Healthy Soybean',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Healthy Strawberry',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites (Two-Spotted Spider Mite)',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Healthy Tomato'
}

disease_model_path = 'model_params.pth'
disease_model = CNN_NeuralNet(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    tr = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = disease_model(tr)
        predicted_index = output.argmax(dim=1).item()
        predicted_class = disease_classes[predicted_index]
        return predicted_class

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict the disease using your prediction function
            predicted_disease = predict(file_path)
            out = disease_names[predicted_disease]
            output = f"The Detected image is of {disease_names[predicted_disease]}"
            pr = f"""
        You are KrushiMitra an expert assistant specializing in agriculture and farming.
        If the plant disease is identified as {out}, provide a detailed and helpful response about effective practices and methods to manage or treat this disease. Suggest suitable fertilizers if any and any relevant preventive measures and treatment options.
        If {out} indicates 'plant is healthy', provide general advice on maintaining plant health and preventing diseases. Your responses should be helpful, clear, and relevant.
        """
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0,
                max_tokens=None,
                timeout=None,
                google_api_key='AIzaSyCQbNkygleMD3b6QI1QFq8-Zr9gpMBAfP4'
            )
            language = session.get('language', 'en')
            translator = Translator()
            llm_out = llm.invoke(pr)
            formatted_output = markdown.markdown(llm_out.content)
            formatted_output= translator.translate(formatted_output, dest=language).text
            return render_template(f'upload{get_language_suffix()}.html', filename=filename, prediction=formatted_output)
        else:
            return redirect(request.url)
    return render_template(f'upload{get_language_suffix()}.html')
          
    
#---------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
