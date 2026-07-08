# 🏗️ Smart Construction Resource Monitoring & Forecasting Platform using BIM–IoT–ML Integration

An AI-powered construction management platform that integrates **Building Information Modeling (BIM)**, **IoT data simulation**, **Machine Learning**, and a **Streamlit dashboard** to forecast construction resource demand and support proactive project planning.

---

## 📌 Project Overview

Construction projects frequently experience delays, cost overruns, and inefficient resource allocation due to fragmented monitoring systems and lack of predictive insights.

This project combines:

- 🏢 BIM (Autodesk Revit)
- 📡 IoT Data Simulation
- 🤖 Machine Learning
- 📊 Streamlit Dashboard

to monitor construction resources and predict the next **7 days of concrete demand**, enabling smarter decision-making and efficient resource management.

---

## 🚀 Features

- ✅ BIM-based quantity extraction using Autodesk Revit
- ✅ Synthetic IoT data generation for 150 construction days
- ✅ Machine Learning based resource forecasting
- ✅ Comparison of Random Forest, XGBoost and LightGBM
- ✅ Interactive Streamlit dashboard
- ✅ 7-Day concrete demand prediction
- ✅ Construction KPI visualization
- ✅ CO₂ emission estimation
- ✅ Labour and weather impact simulation

---

## 🛠 Tech Stack

### BIM
- Autodesk Revit

### Programming
- Python
- Jupyter Notebook

### Libraries

- Panda
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Matplotlib
- Streamlit

### Dataset

- Revit Quantity Take-off Data
- Synthetic IoT Dataset (150 Days)

---

## 📂 Repository Structure

```
smart_resource_forcasting/

│── IOT.ipynb
│── ML.ipynb
│── streamlit.ipynb
│── Revit_Data.csv
│── Training_Data_for_ML.csv
│── README.md
```

---

## 🔄 Workflow

```
Autodesk Revit
        │
        ▼
Material Quantity Extraction
        │
        ▼
IoT Data Simulation
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning Models
(Random Forest | XGBoost | LightGBM)
        │
        ▼
7-Day Resource Forecast
        │
        ▼
Streamlit Dashboard
```

---

## 📈 Machine Learning Models

The project compares three ensemble learning algorithms.

| Model | Speed | Accuracy | Overfitting Risk |
|--------|-------|----------|------------------|
| Random Forest | Medium | Good | Low |
| XGBoost | Slower | Very High | Medium |
| LightGBM | Fastest | Very High | Higher |

LightGBM was selected for deployment because of its high prediction accuracy and excellent computational efficiency.

---

## 📊 Input Parameters

The prediction model uses multiple construction site parameters including:

- Labour Count
- Rain Delay
- Weather Condition
- Weather Intensity
- Truck Travel Time
- Concrete Quality
- Daily Concrete Usage
- CO₂ Emission
- Material Shortage Risk

---

## 📈 Output

The platform predicts:

- Daily Concrete Usage
- Next 7-Day Concrete Demand
- Resource Utilization
- CO₂ Emissions
- Construction KPIs

---

## 📱 Streamlit Dashboard

The dashboard allows users to:

- Upload Revit CSV files
- Simulate site conditions
- Visualize KPIs
- Forecast material demand
- View interactive charts
- Generate a 7-day prediction

---

## 🎯 Applications

- Smart Construction
- BIM-based Planning
- Resource Management
- Project Scheduling
- Construction Analytics
- AI-assisted Decision Support

---

## 🔮 Future Scope

- Live IoT Sensor Integration
- Digital Twin Synchronization
- Mobile Application
- Cloud Deployment
- Real-time Data Streaming
- ESG & Carbon Footprint Prediction
- AI-based Anomaly Detection

---

## 👨‍💻 Mentors

- Yash Kedia
- Bidisha Koley
- Saranksh TR

---

## 👨‍🎓 Mentees

- Piyush Chouhan
- Sreenidhi Golamaru

---

## 📚 References

- BIM Handbook (Eastman et al.)
- XGBoost Research Paper
- LightGBM Research Paper
- Automation in Construction Journal

---

## ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.
