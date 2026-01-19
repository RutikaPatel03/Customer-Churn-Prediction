import tkinter as tk
from tkinter import messagebox
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================
# Load ML Model + Encoders + Feature Names
# ======================
model = pickle.load(open("model/churn_model.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))
feature_names = pickle.load(open("model/feature_names.pkl", "rb"))

# ======================
# GUI Window
# ======================
root = tk.Tk()
root.title("Customer Churn Prediction System")
root.geometry("900x720")
root.resizable(False, False)

# ======================
# Title
# ======================
tk.Label(
    root,
    text="Customer Churn Prediction",
    font=("Arial", 20, "bold"),
    fg="darkblue"
).pack(pady=10)

# ======================
# Input Frame
# ======================
input_frame = tk.Frame(root, bd=2, relief="groove")
input_frame.pack(pady=10, padx=10, fill="x")

# ======================
# Input Fields (Dropdown + Entry)
# ======================
fields = {
    "Gender": ["Male", "Female"],
    "Senior Citizen (0/1)": None,
    "Tenure (Months)": None,
    "Monthly Charges": None,
    "Total Charges": None,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Payment Method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

entries = {}

for label, options in fields.items():
    tk.Label(input_frame, text=label, font=("Arial", 10)).pack()
    if options:
        var = tk.StringVar(value=options[0])
        tk.OptionMenu(input_frame, var, *options).pack(pady=2)
        entries[label] = var
    else:
        entry = tk.Entry(input_frame)
        entry.pack(pady=2)
        entries[label] = entry

# ======================
# Prediction Function
# ======================
def predict_churn():
    try:
        data = {
            "gender": entries["Gender"].get(),
            "SeniorCitizen": int(entries["Senior Citizen (0/1)"].get()),
            "tenure": int(entries["Tenure (Months)"].get()),
            "MonthlyCharges": float(entries["Monthly Charges"].get()),
            "TotalCharges": float(entries["Total Charges"].get()),
            "Contract": entries["Contract"].get(),
            "PaymentMethod": entries["Payment Method"].get()
        }

        input_df = pd.DataFrame([data])

        # Encode categorical
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # Align features
        df_encoded = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            df_encoded[col] = input_df[col] if col in input_df.columns else 0

        prediction = model.predict(df_encoded)

        if prediction[0] == 1:
            messagebox.showwarning("Result", "⚠️ Customer is likely to CHURN")
        else:
            messagebox.showinfo("Result", "✅ Customer is NOT likely to churn")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ======================
# Predict Button
# ======================
tk.Button(
    root,
    text="Predict Churn",
    font=("Arial", 14),
    bg="#003366",
    fg="white",
    command=predict_churn
).pack(pady=10)

# ======================
# Graph Section
# ======================
graph_frame = tk.Frame(root)
graph_frame.pack(pady=10)

def show_graphs():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

    fig = plt.Figure(figsize=(12, 4))
    
    # ---- Pie Chart ----
    ax1 = fig.add_subplot(131)
    df["Churn"].value_counts().plot.pie(
        autopct="%1.1f%%",
        startangle=90,
        colors=["green", "red"],
        ax=ax1
    )
    ax1.set_title("Churn Distribution")
    ax1.set_ylabel("")

    # ---- Histogram ----
    ax2 = fig.add_subplot(132)
    ax2.hist(df["MonthlyCharges"], bins=20)
    ax2.set_title("Monthly Charges")
    ax2.set_xlabel("Charges")

    # ---- Bar Chart ----
    ax3 = fig.add_subplot(133)
    churn_counts = df.groupby("Churn")["tenure"].mean()
    ax3.bar(churn_counts.index, churn_counts.values)
    ax3.set_title("Avg Tenure vs Churn")

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

tk.Button(
    root,
    text="Show Graphs",
    bg="green",
    fg="white",
    font=("Arial", 12),
    command=show_graphs
).pack(pady=10)

root.mainloop()
