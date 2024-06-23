from flask import Flask, jsonify, render_template, request
import pickle

app = Flask(__name__)

crops = {
    1: 'Rice',
    2: 'Maize',
    3: 'ChickPea',
    4: 'KidneyBeans',
    5: 'PigeonPeas',
    6: 'MothBeans',
    7: 'MungBean',
    8: 'Blackgram',
    9: 'Lentil',
    10: 'Pomegranate',
    11: 'Banana',
    12: 'Mango',
    13: 'Grapes',
    14: 'Watermelon',
    15: 'Muskmelon',
    16: 'Apple',
    17: 'Orange',
    18: 'Papaya',
    19: 'Coconut',
    20: 'Cotton',
    21: 'Jute',
    22: 'Coffee'
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/me")
def me():
    return "Hello, this is me"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        nitrogen = float(request.form.get("nitrogen"))
        potassium = float(request.form.get("potassium"))
        phosphorus = float(request.form.get("phosphorus"))
        rainfall = float(request.form.get("rainfall"))
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        ph_value = float(request.form.get("ph_value"))
        
        with open("model_one.pkl", "rb") as f:
            model = pickle.load(f)
        
        result = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
        recommended_crop = crops[result[0]]
        
        return render_template("home.html", crop=recommended_crop)
    except ValueError:
        return "Invalid input. Please enter valid numbers.", 400
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
