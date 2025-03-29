#✅ Step 1: Data Collection & Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from google.colab import files

#uploaded = files.upload()
#uploaded.keys()
df = pd.read_csv(io.BytesIO(uploaded['Churn Modeling.csv']))
df.head()
df.info()
df.isnull().sum()
df.shape
df.describe()

#✅ Step 2: Exploratory Data Analysis (EDA)
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.head()


sns.countplot(x=df["Exited"], palette=["green", "red"])

df.hist(figsize=(12, 8))
plt.figure(figsize=(10,6))


plt.show()
sns.boxplot(x="Exited", y="Age", data=df, palette=["green", "red"])
sns.countplot(x="Geography", hue="Exited", data=df)
sns.countplot(x="Gender", hue="Exited", data=df)

sns.boxplot(x=df["Balance"])
from sklearn.tree import DecisionTreeClassifier #LIBRARY, MODULE , CLASS
model = DecisionTreeClassifier()
model.fit(X,Y)
predictions= model.predict([[21,1], [22, 0]])
df["Age_Group"] = pd.cut(df["Age"], bins=[18, 30, 50, 100], labels=["Young", "Middle", "Senior"])
sns.countplot(x="Age_Group", hue="Exited", data=df)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#80 percent for training
#20 percent for test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
score = accuracy_score(Y_test, predictions)

predictions
score

#Convert Categorical Features to Numeric
from sklearn.preprocessing import LabelEncoder  

# Convert categorical columns  
df["Geography"] = LabelEncoder().fit_transform(df["Geography"])  
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])  
#📌 This changes "Male/Female" to 0/1 and "France/Germany/Spain" to numerical values.  
#🔥 1.4 Select Features & Target Variable

 #   X (Features): All columns except "Exited".

  #  y (Target): "Exited" (1 = Churned, 0 = Not Churned).

X = df.drop(columns=["Exited"])  
y = df["Exited"]  


#🔥 1.5 Split Data into Training & Testing Sets

#We split 80% for training & 20% for testing.
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  
#📌 Why? This ensures that both classes (Churn/Not Churn) are evenly distributed in train & test sets.

 sns.countplot(x=y)  
plt.title("Churn Distribution")  
plt.show()


# Age vs. Churn  
sns.boxplot(x=y, y=df["Age"])  
plt.title("Age vs Churn")  
plt.show()

# Credit Score vs. Churn  
sns.boxplot(x=y, y=df["CreditScore"])  
plt.title("Credit Score vs Churn")  
plt.show()


from sklearn.preprocessing import LabelEncoder  

# Apply Label Encoding
label_enc = LabelEncoder()
df["Surname"] = label_enc.fit_transform(df["Surname"])  
df["Age_Group"] = label_enc.fit_transform(df["Age_Group"])
df["Geography"] = label_enc.fit_transform(df["Geography"])  
df["Gender"] = label_enc.fit_transform(df["Gender"])
df.dtypes
for col in df.columns:
    if df[col].dtype == 'object':  # Find text columns
# Find which column contains 'Walton'
      for col in df.columns:
        if df[col].dtype == 'object':  # Checking only text columns
           print(f"Column '{col}' has text data. Unique values: {df[col].unique()[:5]}")

from sklearn.preprocessing import LabelEncoder  

encoder = LabelEncoder()
df["Geography"] = encoder.fit_transform(df["Geography"])  
df["Gender"] = encoder.fit_transform(df["Gender"])  

df[df.apply(lambda row: row.astype(str).str.contains('Walton', na=False).any(), axis=1)]


df[df.apply(lambda row: row.astype(str).str.contains('Walton', na=False).any(), axis=1)]

print(X_train.dtypes)
print(X_test.dtypes)


print(X_train["Age_Group"].isna().sum())
print(X_test["Age_Group"].isna().sum())

from sklearn.preprocessing import LabelEncoder  

encoder = LabelEncoder()
X_train["Age_Group"] = encoder.fit_transform(X_train["Age_Group"])  
X_test["Age_Group"] = encoder.transform(X_test["Age_Group"])  
print(X_train.dtypes)
print(X_test.dtypes)

log_reg.fit(X_train, y_train)


#🔥 3.1 Train Logistic Regression

from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  

log_reg = LogisticRegression()  
log_reg.fit(X_train, y_train)  
y_pred_log = log_reg.predict(X_test)  

# Evaluate  
accuracy_score(y_test, y_pred_log)

#🔥 3.2 Train Random Forest

from sklearn.ensemble import RandomForestClassifier  

rf = RandomForestClassifier(n_estimators=100, random_state=42)  
rf.fit(X_train, y_train)  
y_pred_rf = rf.predict(X_test)  

accuracy_score(y_test, y_pred_rf)

#🔥 3.3 Train XGBoost

from xgboost import XGBClassifier  

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")  
xgb.fit(X_train, y_train)  
y_pred_xgb = xgb.predict(X_test)  

accuracy_score(y_test, y_pred_xgb)

import pickle

# Suppose tumhara trained model ye hai
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save the trained model
with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")



!pip install flask
#2️⃣ Create app.py (Flask API)

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get input data
    features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
    prediction = model.predict(features)  # Predict churn
    return jsonify({"Churn Prediction": int(prediction[0])})  # Return result

if __name__ == "__main__":
    app.run(debug=True)

#📌 Save this file as app.py and run it.
#This will start a local API where you can send customer data and get churn predictions!

Run Jupyter Notebooks for EDA & Model Training

Deploy API using python app.py

Automate Emails using send_email.py

Monitor in Power BI 🎯
