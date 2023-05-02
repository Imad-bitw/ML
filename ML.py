import streamlit as st
import streamlit.components.v1 as stc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

# File Processing Pkgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber


def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def read_pdf_with_pdfplumber(file):
	with pdfplumber.open(file) as pdf:
	    page = pdf.pages[0]
	    return page.extract_text()

# import fitz  # this is pymupdf

# def read_pdf_with_fitz(file):
# 	with fitz.open(file) as doc:
# 		text = ""
# 		for page in doc:
# 			text += page.getText()
# 		return text 

# Fxn
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 



def main():
	st.title("Machine Learning Project")
	st.write("""
	#Supervised by : TALI Abdelhak\n
	Prepared by : EL KHLIFI Imad
	""")
	menu = ["Home","Dataset","DocumentFiles","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
		if image_file is not None:
		
			# To See Details
			# st.write(type(image_file))
			# st.write(dir(image_file))
			file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
			st.write(file_details)

			img = load_image(image_file)
			st.image(img,width=250,height=250)


	elif choice == "Dataset":
		st.subheader("Dataset")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if st.button("Process"):
			if data_file is not None:
				file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
				st.write(file_details)
				df = pd.read_csv(data_file)
				st.dataframe(df)
				X = df.iloc[:,[2,3]]
				Y = df.iloc[:,4]
				st.write(X)
				st.write(Y)
				# Review salary frequencies
				X_Train, X_Test,Y_Train, Y_Test = train_test_split(X,Y, test_size=0.25, random_state=0)
				#Training=print("Training data :", X_Train.shape)
				#Trainingtest=print("Training data :", X_Test.shape)
				st.write("Training data :",X_Train.shape)
				st.write("Training data :", X_Test.shape)
				sc_X = StandardScaler()
				X_Train = sc_X.fit_transform(X_Train)
				X_Test = sc_X.transform(X_Test)
				classifier = SVC(kernel = 'linear', random_state=0)
				classifier.fit(X_Train, Y_Train)
				######### Predicting the test set results
				Y_Pred = classifier.predict(X_Test)
				st.write(Y_Pred)
				st.write('Accurancy Score : With Linear Kernel',metrics.accuracy_score(Y_Test, Y_Pred))
				classifier = SVC(kernel = 'rbf')
				classifier.fit(X_Train, Y_Train)

				####### Predicting the test set results
				Y_Pred = classifier.predict(X_Test)
				st.write('Accurancy Score : With default Kernel',metrics.accuracy_score(Y_Test, Y_Pred))
				classifier = SVC(kernel = 'rbf',gamma=15,C=7, random_state=0)
				classifier.fit(X_Train, Y_Train)

				####### Predicting the test set results
				Y_Pred = classifier.predict(X_Test)
				st.write('Accurancy Score on Test Data : With default Kernel',metrics.accuracy_score(Y_Test, Y_Pred))
				svc=SVC(kernel='poly',degree = 4)
				svc.fit(X_Train, Y_Train)

				y_pred= svc.predict(X_Test)
				st.write('Accurancy Score : With poly Kernel and degree',metrics.accuracy_score(Y_Test, Y_Pred))
				plt.scatter(X_Train[:,0], X_Train[:,1], c=Y_Train)
				plt.xlabel('Age')
				plt.ylabel('Estimated Salary')
				plt.title('Test Data')
				st.pyplot()	
				plt.scatter(X_Test[:,0], X_Test[:,1], c=Y_Test)
				plt.xlabel('Age')
				plt.ylabel('Estimated Salary')
				plt.title('Test Data')
				st.pyplot()

				classifier = SVC(kernel = 'linear', random_state=0)
				classifier.fit(X_Train, Y_Train)
				######### Predicting the test set results
				Y_Pred = classifier.predict(X_Test)
				#plot data points
				plt.scatter(X_Test[:,0], X_Test[:,1], c=Y_Test)
				###### Ceate the hyperplane
				w = classifier.coef_[0]
				a = -w[0] / w[1]
				xx = np.linspace(-2.5,2.5)
				yy = a * xx - (classifier.intercept_[0]) / w[1]

				#### Plot the hyperplane
				plt.plot(xx, yy)
				plt.axis("off")
				st.pyplot()
				





	elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")
		docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
		if st.button("Process"):
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)
				# Check File Type
				if docx_file.type == "text/plain":
					# raw_text = docx_file.read() # read as bytes
					# st.write(raw_text)
					# st.text(raw_text) # fails
					st.text(str(docx_file.read(),"utf-8")) # empty
					raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
					# st.text(raw_text) # Works
					st.write(raw_text) # works
				elif docx_file.type == "application/pdf":
					# raw_text = read_pdf(docx_file)
					# st.write(raw_text)
					try:
						with pdfplumber.open(docx_file) as pdf:
						    page = pdf.pages[0]
						    st.write(page.extract_text())
					except:
						st.write("None")
					    
					
				elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				# Use the right file processor ( Docx,Docx2Text,etc)
					raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
					st.write(raw_text)

	else:
		st.subheader("About")
		st.info("Built with Streamlit")
		st.info("imadelkhlifi@gmail.com")
		st.text("EL KHLIFI Imad")


if __name__ == '__main__':
	main()
