import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

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
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

def predict_outcome():
    try:
        inputs = [
            float(entry_age.get()),
            float(entry_glucose.get()),
            float(entry_bmi.get()),
            float(entry_pressure.get()),
            float(entry_skin_thickness.get()),
            float(entryinsulin.get()),
            float(entry_pedigree.get()),
            float(entry_age.get())
        ]
        inputs_scaled = scaler.transform([inputs])
        prediction = rf_model.predict(inputs_scaled)
        result = "Diabetes Positive" if prediction == 1 else "Diabetes Negative"
        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")


root = tk.Tk()
root.title("Diabetes Prediction")

label_age = tk.Label(root, text="Age")
label_age.pack()
entry_age = tk.Entry(root)
entry_age.pack()

label_glucose = tk.Label(root, text="Glucose Level")
label_glucose.pack()
entry_glucose = tk.Entry(root)
entry_glucose.pack()

label_bmi = tk.Label(root, text="BMI")
label_bmi.pack()
entry_bmi = tk.Entry(root)
entry_bmi.pack()

label_pressure = tk.Label(root, text="Blood Pressure")
label_pressure.pack()
entry_pressure = tk.Entry(root)
entry_pressure.pack()

label_skin_thickness = tk.Label(root, text="Skin Thickness")
label_skin_thickness.pack()
entry_skin_thickness = tk.Entry(root)
entry_skin_thickness.pack()

label_insulin = tk.Label(root, text="Insulin")
label_insulin.pack()
entryinsulin = tk.Entry(root)
entryinsulin.pack()

label_pedigree = tk.Label(root, text="Diabetes Pedigree Function")
label_pedigree.pack()
entry_pedigree = tk.Entry(root)
entry_pedigree.pack()

predict_button = tk.Button(root, text="Predict", command=predict_outcome)
predict_button.pack()

root.mainloop()
