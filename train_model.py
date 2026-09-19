import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Train Model")

def get_risk(aqi):
    if aqi <= 50:
        return "Healthy"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # rename columns (simple)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "city": "City",
        "factory": "Factory_Type",
        "aqi": "AQI",
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO2",
        "so2": "SO2",
        "co": "CO"
    }, inplace=True)

    st.write(df.head())

    needed = ["City","Factory_Type","AQI","PM2.5","PM10","NO2","SO2","CO"]

    if all(col in df.columns for col in needed):

        df = df[needed].dropna()
        df["Risk"] = df["AQI"].apply(get_risk)
        st.write(df["Risk"].value_counts())
        df = df.sample(frac=1, random_state=42)

        st.write("Rows:", len(df))

        if st.button("Train Model"):

            # encoding
            cities = {c:i for i,c in enumerate(df["City"].unique())}
            factories = {f:i for i,f in enumerate(df["Factory_Type"].unique())}

            df["City_Code"] = df["City"].map(cities)
            df["Factory_Code"] = df["Factory_Type"].map(factories)

            X = df[["Factory_Code","PM2.5","PM10","NO2","SO2","CO"]]
            y = df["Risk"]

            X_train,X_test,y_train,y_test = train_test_split(
                X,y,test_size=0.2,random_state=42,stratify=y
                )
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10, min_samples_split=5, random_state=42
                )
            model.fit(X_train,y_train)

            # save
            pickle.dump(model, open("model.pkl","wb"))
            pickle.dump(cities, open("city_mapping.pkl","wb"))
            pickle.dump(factories, open("factory_mapping.pkl","wb"))
            pickle.dump(df, open("training_dataset.pkl","wb"))

            # evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model Trained ✅ - Accuracy: {round(accuracy*100,2)}%")

    else:
        st.error("Dataset columns missing")