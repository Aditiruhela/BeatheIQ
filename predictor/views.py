import os
import requests
import joblib
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# LOAD ML FILES (LOAD ONLY ONCE)
# =========================================================
try:
    model = joblib.load(os.path.join(BASE_DIR, 'predictor/ml/model.pkl'))
    city_map = joblib.load(os.path.join(BASE_DIR, 'predictor/ml/city_mapping.pkl'))
    factory_map = joblib.load(os.path.join(BASE_DIR, 'predictor/ml/factory_mapping.pkl'))
except:
    model = None
    city_map = {}
    factory_map = {}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_subindex(cp, breakpoints):
    for bp in breakpoints:
        blo, bhi, ilo, ihi = bp
        if blo <= cp <= bhi:
            return ((ihi - ilo) / (bhi - blo)) * (cp - blo) + ilo
    return 500


def calc_aqi(pm25, pm10, no2, so2, co):
    pm25_bp = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ]

    pm10_bp = [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500)
    ]

    no2_bp = [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 600, 401, 500)
    ]

    so2_bp = [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500)
    ]

    co_bp = [
        (0, 1, 0, 50),
        (1.1, 2, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500)
    ]

    sub_indices = [
        get_subindex(pm25, pm25_bp),
        get_subindex(pm10, pm10_bp),
        get_subindex(no2, no2_bp),
        get_subindex(so2, so2_bp),
        get_subindex(co, co_bp)
    ]

    return round(max(sub_indices), 2)


def risk_label(aqi):
    if aqi <= 50:
        return "Good / Green"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


def diseases(risk):
    if risk == "Good / Green":
        return ["No major health risk", "Air quality is good"]
    elif risk == "Satisfactory":
        return ["Minor breathing discomfort for sensitive people"]
    elif risk == "Moderate":
        return ["Cough", "Throat irritation", "Breathing discomfort"]
    elif risk == "Poor":
        return ["Asthma attacks", "Bronchitis", "Chest tightness"]
    elif risk == "Very Poor":
        return ["Respiratory illness", "Reduced lung function", "Heart risk"]
    return ["Severe lung problems", "Heart diseases", "Serious respiratory illness"]


def pollutant_effect(pollutant):
    effects = {
        "PM2.5": "Deep lung penetration causes lung damage",
        "PM10": "Respiratory irritation and coughing",
        "NO2": "Lung inflammation and asthma",
        "SO2": "Breathing issues and throat irritation",
        "CO": "Reduces oxygen in blood, harmful for heart"
    }
    return effects.get(pollutant, "No data available")


def precautions_by_disease(disease):
    data = {
        "Asthma attacks": {
            "children": [
                "Avoid outdoor play during high AQI",
                "Wear mask while going outside",
                "Keep inhaler ready if prescribed"
            ],
            "adults": [
                "Avoid smoking and dust exposure",
                "Do not exercise in polluted areas",
                "Wear N95 mask outdoors"
            ],
            "elderly": [
                "Stay indoors during poor AQI",
                "Keep inhaler and medicines ready",
                "Use air purifier if available"
            ]
        },

        "Bronchitis": {
            "children": [
                "Avoid cold drinks",
                "Drink warm water",
                "Keep away from dust and smoke"
            ],
            "adults": [
                "Avoid smoking",
                "Stay hydrated",
                "Wear mask in polluted areas"
            ],
            "elderly": [
                "Take steam regularly",
                "Avoid cold exposure",
                "Consult doctor if cough increases"
            ]
        },

        "Respiratory illness": {
            "children": [
                "Avoid outdoor activities",
                "Drink clean warm fluids",
                "Use mask outside"
            ],
            "adults": [
                "Avoid traffic-heavy areas",
                "Use air purifier if possible",
                "Drink plenty of water"
            ],
            "elderly": [
                "Stay indoors",
                "Take steam inhalation",
                "Seek medical help if breathing worsens"
            ]
        },

        "Heart diseases": {
            "children": [
                "Avoid polluted outdoor exposure",
                "Eat healthy food",
                "Stay hydrated"
            ],
            "adults": [
                "Avoid stress and smoking",
                "Limit outdoor activity in poor AQI",
                "Monitor chest discomfort"
            ],
            "elderly": [
                "Avoid morning walks in polluted air",
                "Take regular medicines",
                "Consult doctor if discomfort occurs"
            ]
        },

        "Severe lung problems": {
            "children": [
                "Avoid outdoor exposure completely",
                "Use mask if going outside",
                "Seek doctor if coughing starts"
            ],
            "adults": [
                "Avoid smoke and industrial areas",
                "Use N95 mask",
                "Stay indoors as much as possible"
            ],
            "elderly": [
                "Use inhaler / oxygen support if prescribed",
                "Do not go outside in severe AQI",
                "Keep emergency contact ready"
            ]
        }
    }

    return data.get(disease, {
        "children": ["Drink clean water", "Wear mask outside"],
        "adults": ["Avoid pollution exposure", "Stay hydrated"],
        "elderly": ["Stay indoors", "Avoid polluted air"]
    })


# =========================================================
# HOME PAGE
# =========================================================
def home(request):
    return render(request, 'predictor/home.html')


# =========================================================
# AQI + LIVE CITY DATA PAGE
# =========================================================
def predict(request):
    if request.method == "POST":
        city = request.POST.get('city')

        AQICN_TOKEN = "24c1c414f9834857cce598e4fc8e3b36e1b7edfc"
        aqi_url = f"https://api.waqi.info/feed/{city.lower()}/?token={AQICN_TOKEN}"

        try:
            res = requests.get(aqi_url).json()
            real_aqi = res['data']['aqi'] if res.get('status') == 'ok' else None
        except:
            real_aqi = None

        city_coords = {
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Lucknow": (26.8467, 80.9462),
            "Bhopal": (23.2599, 77.4126),
            "Jaipur": (26.9124, 75.7873),
            "Kolkata": (22.5726, 88.3639),
        }

        lat, lon = city_coords.get(city, (None, None))
        pm25 = pm10 = no2 = so2 = co = None

        if lat and lon:
            try:
                WEATHER_KEY = "196ec8ac4bb9cd0332a464733295baf4"
                url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_KEY}"
                weather = requests.get(url).json()

                comp = weather['list'][0]['components']
                pm25 = comp.get('pm2_5')
                pm10 = comp.get('pm10')
                no2 = comp.get('no2')
                so2 = comp.get('so2')
                co = comp.get('co')
            except:
                pass

        return render(request, 'predictor/predict.html', {
            'city': city,
            'aqi': real_aqi,
            'pm25': pm25,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'co': co,
        })

    return render(request, 'predictor/predict.html')


# =========================================================
# ML + HEALTH ANALYSIS PAGE
# =========================================================
def disease(request):
    result = None

    if request.method == "POST":
        try:
            city = request.POST.get("city")
            factory = request.POST.get("factory")

            pm25 = float(request.POST.get("pm25"))
            pm10 = float(request.POST.get("pm10"))
            no2 = float(request.POST.get("no2"))
            so2 = float(request.POST.get("so2"))
            co = float(request.POST.get("co"))

            # ---------------- REAL AQI ----------------
            aqi = calc_aqi(pm25, pm10, no2, so2, co)

            # ---------------- ML PREDICTION ----------------
            ml_prediction = "Model not available"

            if model:
                try:
                    city_code = city_map.get(city, 0)
                    factory_code = factory_map.get(factory, 0)
                    features = np.array([[city_code, factory_code, aqi, pm25, pm10, no2, so2, co]])
                    ml_prediction = model.predict(features)[0]
                except:
                    ml_prediction = "Prediction Error"

            # ---------------- DOMINANT POLLUTANT ----------------
            values = {
                "PM2.5": pm25,
                "PM10": pm10,
                "NO2": no2,
                "SO2": so2,
                "CO": co
            }

            limits = {
                "PM2.5": 30,
                "PM10": 50,
                "NO2": 40,
                "SO2": 40,
                "CO": 4
            }

            harm_scores = {k: values[k] / limits[k] for k in values}
            dominant = max(harm_scores, key=harm_scores.get)

            disease_list = diseases(risk_label(aqi))
            main_disease = disease_list[0] if disease_list else "General Respiratory Risk"

            # ---------------- FINAL RESULT ----------------
            result = {
                "aqi": round(aqi, 2),
                "risk": risk_label(aqi),
                "ml": ml_prediction,
                "dominant": dominant,
                "effect": pollutant_effect(dominant),
                "diseases": disease_list,
                "precautions": precautions_by_disease(main_disease),
                "green": aqi <= 50
            }

        except Exception as e:
            print("Disease Prediction Error:", e)

    return render(request, 'predictor/disease.html', {
        "result": result,
        "cities": city_map.keys(),
        "factories": factory_map.keys()
    })


@csrf_exempt
def api_predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requests only"}, status=405)

    try:
        payload = json.loads(request.body)
        city = payload.get("city", "")
        factory = payload.get("factory", "")
        pollutants = {
            "PM2.5": float(payload["pm25"]),
            "PM10": float(payload["pm10"]),
            "NO2": float(payload["no2"]),
            "SO2": float(payload["so2"]),
            "CO": float(payload["co"]),
        }

        aqi = calc_aqi(*pollutants.values())
        risk = risk_label(aqi)
        limits = {"PM2.5": 30, "PM10": 50, "NO2": 40, "SO2": 40, "CO": 4}
        dominant = max(
            pollutants,
            key=lambda pollutant: pollutants[pollutant] / limits[pollutant]
        )

        prediction = "Model not available"
        if model:
            city_code = city_map.get(city, 0)
            factory_code = factory_map.get(factory, 0)
            features = np.array([[city_code, factory_code, aqi, *pollutants.values()]])
            try:
                prediction = model.predict(features)[0]
            except (TypeError, ValueError):
                prediction = "Prediction Error"

        return JsonResponse({
            "city": city,
            "aqi": round(aqi, 2),
            "risk": risk,
            "prediction": str(prediction),
            "dominant": dominant,
            "effect": pollutant_effect(dominant),
            "diseases": diseases(risk),
            "green": aqi <= 50,
        })
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return JsonResponse({"error": f"Invalid prediction data: {error}"}, status=400)


# =========================================================
# CITY COMPARISON PAGE
# =========================================================
def result(request):
    AQICN_TOKEN = "24c1c414f9834857cce598e4fc8e3b36e1b7edfc"
    cities = ["Delhi", "Mumbai", "Lucknow", "Bhopal", "Jaipur", "Kolkata"]

    city_data = []

    for city in cities:
        try:
            url = f"https://api.waqi.info/feed/{city.lower()}/?token={AQICN_TOKEN}"
            res = requests.get(url).json()
            aqi = res['data']['aqi'] if res.get('status') == 'ok' else 999
        except:
            aqi = 999

        label = risk_label(aqi)

        city_data.append({
            "name": city,
            "aqi": aqi,
            "label": label
        })

    best = min(city_data, key=lambda x: x["aqi"])

    for c in city_data:
        c["is_best"] = c["aqi"] == best["aqi"]

    return render(request, "predictor/result.html", {
        "city_data": city_data,
        "green_city": best["name"]
    })


# =========================================================
# ABOUT PAGE
# =========================================================
def about(request):
    return render(request, 'predictor/about.html')


# =========================================================
# CONTACT PAGE
# =========================================================
def contact(request):
    return render(request, 'predictor/contact.html')