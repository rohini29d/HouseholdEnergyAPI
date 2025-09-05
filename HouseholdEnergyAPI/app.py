from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import json
import os
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()
templates = Jinja2Templates(directory="templates")

USERS_FILE = "users.json"
UPLOADED_FILE = "uploaded_data.csv"
MODEL_FILE = "model.pkl"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

# ---------- ML FUNCTIONS ----------
def train_model():
    df = pd.read_csv(UPLOADED_FILE, parse_dates=[['Date', 'Time']])
    df.set_index('Date_Time', inplace=True)
    df['hour'] = df.index.hour

    q_low = df['Global_active_power'].quantile(0.01)
    q_hi = df['Global_active_power'].quantile(0.99)
    df = df[(df['Global_active_power'] > q_low) & (df['Global_active_power'] < q_hi)]

    features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour']
    X = df[features]
    y = df['Global_active_power']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return mse, r2

def predict_sample():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(UPLOADED_FILE, parse_dates=[['Date', 'Time']])
    df['hour'] = df['Date_Time'].dt.hour

    features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour']
    sample = df[features].sample(1).to_dict(orient="records")[0]
    prediction = model.predict(pd.DataFrame([sample]))[0]

    return round(prediction, 3), sample

# ---------- AUTH ROUTES ----------
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "msg": ""})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    if username in users and users[username] == password:
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "msg": "❌ Invalid login."})

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "msg": ""})

@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    if username in users:
        return templates.TemplateResponse("register.html", {"request": request, "msg": "❌ Username already exists."})
    users[username] = password
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    return RedirectResponse("/", status_code=302)

# ---------- DASHBOARD ----------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    csv_uploaded = os.path.exists(UPLOADED_FILE)
    model_trained = os.path.exists(MODEL_FILE)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "csv_uploaded": csv_uploaded,
        "model_trained": model_trained,
        "mse": None,
        "r2": None,
        "prediction": None,
        "auto_input": None,
        "graph_url": None,
        "popup_message": None
    })

@app.post("/upload_csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    with open(UPLOADED_FILE, "wb") as f:
        f.write(contents)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "csv_uploaded": True,
        "model_trained": os.path.exists(MODEL_FILE),
        "mse": None,
        "r2": None,
        "prediction": None,
        "auto_input": None,
        "graph_url": None,
        "popup_message": "✅ File uploaded successfully!"
    })

@app.post("/train_model")
def train(request: Request):
    mse, r2 = train_model()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "csv_uploaded": True,
        "model_trained": True,
        "mse": mse,
        "r2": r2,
        "prediction": None,
        "auto_input": None,
        "graph_url": None,
        "popup_message": None
    })

@app.post("/predict")
def show_prediction_form(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "csv_uploaded": True,
        "model_trained": True,
        "show_manual_form": True,
        "mse": None,
        "r2": None,
        "prediction": None,
        "auto_input": None,
        "graph_url": None,
        "popup_message": None
    })

@app.post("/manual_predict")
def manual_predict(
    request: Request,
    global_reactive_power: float = Form(...),
    voltage: float = Form(...),
    global_intensity: float = Form(...),
    sub_metering_1: int = Form(...),
    sub_metering_2: int = Form(...),
    sub_metering_3: int = Form(...),
    hour: int = Form(...)
):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    sample = {
        "Global_reactive_power": global_reactive_power,
        "Voltage": voltage,
        "Global_intensity": global_intensity,
        "Sub_metering_1": sub_metering_1,
        "Sub_metering_2": sub_metering_2,
        "Sub_metering_3": sub_metering_3,
        "hour": hour
    }

    prediction = model.predict(pd.DataFrame([sample]))[0]

    # Generate comparison graph
    df = pd.read_csv(UPLOADED_FILE)
    avg = df["Global_active_power"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(["Predicted"], [prediction], color="#007acc", label="Predicted")
    plt.bar(["Average"], [avg], color="#bbb", label="Average")
    plt.ylabel("kW")
    plt.title("Predicted vs Average Power")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "csv_uploaded": True,
        "model_trained": True,
        "mse": None,
        "r2": None,
        "prediction": round(prediction, 3),
        "auto_input": sample,
        "graph_url": graph_url,
        "popup_message": None,
        "show_manual_form": False
    })
