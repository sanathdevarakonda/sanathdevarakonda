import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle


class DataFrame_Loader():

    
    def __init__(self):
        
        print("Loading DataFrame")
        
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df
    
class EDA_Dataframe_Analysis():

    def show_columns(self,x):
           return x.columns
  
    def Show_CountPlot(self,x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x,ax=ax)

classifier = pickle.load(open('classifierdt.sav','rb'))        

def prediction(voice_mail_plan,voice_mail_messages,day_mins,evening_mins,night_mins,international_mins,customer_service_calls,international_plan,day_charge,evening_charge,night_charge,international_charge,total_charge):  

    prediction = classifier.predict(
        [[voice_mail_plan,voice_mail_messages,day_mins,evening_mins,night_mins,international_mins,customer_service_calls,international_plan,day_charge,evening_charge,night_charge,international_charge,total_charge]])
    print(prediction)
    return prediction

def main():
 
    st.title("Machine Learning Project")

    activities = ["EDA","Model Building for Classification Problem"]

    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == 'Model Building for Classification Problem':

       st.title("Telecom Churn Prediction")

       html_temp = """
       <div style ="background-color:yellow;padding:13px">
       <h1 style ="color:black;text-align:center;">Streamlit Churn Prediction Classifier ML App </h1>
       </div>
       """

       st.markdown(html_temp, unsafe_allow_html = True)
       voice_mail_plan = st.selectbox('Voicemail plan',[0,1])
       voice_mail_messages=st.slider('Voicemail messages',0,100)
       day_mins = st.slider('Day mins',0,500) 
       evening_mins = st.slider('Evening mins',0,500)
       night_mins = st.slider('Night mins',0,500)
       international_mins = st.slider('International mins',0,100)
       customer_service_calls = st.number_input('Customer Service Calls', 0,20)
       international_plan = st.selectbox('International plan',[0,1])
       day_charge = st.slider('Day Charge',0,100) 
       evening_charge = st.slider('Evening Charge',0,100)
       night_charge = st.slider('Night Charge',0,100)
       international_charge= st.slider('International Charge',0,100)
       total_charge = st.slider('Total Charge',0,200)
       result =""  
       if st.button("Predict"):
          result = prediction(voice_mail_plan,voice_mail_messages,day_mins,evening_mins,night_mins,international_mins,customer_service_calls,international_plan,day_charge,evening_charge,night_charge,international_charge,total_charge)  

       st.success('The output is {}'.format(result))

    else: 
     
         data = st.file_uploader("Upload a Dataset", type=["csv"])
         if data is not None:
            df=load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")

         all_columns_names = dataframe.show_columns(df)         
         selected_columns_names = st.selectbox("Select Columns CountPlot ",all_columns_names)
         if st.checkbox("Show CountPlot for Selected variable"):
           st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
           st.pyplot() 

if __name__ == '__main__':
   load = DataFrame_Loader()
   dataframe = EDA_Dataframe_Analysis()
   main()
   













     
