from numpy.core.numeric import True_
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score


def main():
	st.title("Machine Learning Project")
	st.write("""
	#Supervised by : TALI Abdelhak\n
	Prepared by : EL KHLIFI Imad
	""")
	st.sidebar.title("Menu")
	#st.sidebar.markdown("Let's start with binary classification!!")
if __name__ == '__main__':
	main()

@st.cache(persist= True)
def load():
	data= pd.read_csv("data.csv")
	#label= LabelEncoder()
	#for i in data.columns:
		#data[i] = label.fit_transform(data[i])
	return data
df = load()
if st.sidebar.checkbox("Display data", False):
	st.subheader("Dataset")
	st.write(df)

@st.cache(persist=True)
def split(df):
	x = df.iloc[:,[2,3]]
	y = df.iloc[:,4]
	x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
	return x_train, x_test, y_train, y_test
	x_train, x_test, y_train, y_test = split(df)

def plot_metrics(metrics_list):
	if "Confusion Matrix" in metrics_list:
		st.subheader("Confusion Matrix")
		ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=   class_names)
		st.pyplot()
	if "ROC Curve" in metrics_list:
		st.subheader("ROC Curve")
		RocCurveDisplay.from_estimator(model, x_test, y_test)
		st.pyplot()
	if "Precision-Recall Curve" in metrics_list:
		st.subheader("Precision-Recall Curve")
		PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
		st.pyplot()
class_names = ["Age", "Estimated Salary"]

st.sidebar.subheader("Choose classifier")

classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
if classifier == "Support Vector Machine (SVM)":
	st.sidebar.subheader("Hyperparameters")
	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
	kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
	gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

if st.sidebar.button("Classify", key="classify"):
	st.subheader("Support Vector Machine (SVM) results")
	model = SVC(C=C, kernel=kernel, gamma=gamma)
	x = df.iloc[:,[2,3]]
	y = df.iloc[:,4]
	x_train, x_test,y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
	model.fit(x_train, y_train)
	accuracy = model.score(x_test, y_test)
	y_pred = model.predict(x_test)
	st.write("Accuracy: ", accuracy.round(2))
	st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
	st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3)) 
	plot_metrics(metrics)

if classifier == "Logistic Regression":
	st.sidebar.subheader("Hyperparameters")
	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
	max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
	metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
	if st.sidebar.button("Classify", key="classify"):
		st.subheader("Logistic Regression Results")
		model = LogisticRegression(C=C, max_iter=max_iter)
		model.fit(x_train, y_train)
		accuracy = model.score(x_test, y_test)
		y_pred = model.predict(x_test)
        
		st.write("Accuracy: ", accuracy.round(2))
		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
		plot_metrics(metrics)