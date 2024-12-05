import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st


df = pd.read_csv(r'C:\Users\Sara Hanif\Desktop\project AI\diabetes.csv')

df.fillna(df.mean(), inplace=True)


X = df.drop(columns=['Outcome'])  
y = df['Outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
st.write(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

st.title("Diabetes Prediction")


pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=90)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=25)


if st.button("Predict"):
    try:

        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
   
        inputs_scaled = scaler.transform([inputs])
        
      
        prediction = rf_model.predict(inputs_scaled)
        

        result = "Diabetes Positive" if prediction == 1 else "Diabetes Negative"
        st.success(f"Prediction: {result}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

