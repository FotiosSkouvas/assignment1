import streamlit as st
import klib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dabl import plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Statement of our multi-element containers
header = st.container()
dataset = st.container()
data_overview = st.container()
data_visualization = st.container()
data_quering = st.container()
data_modeling = st.container()

#Introduction
with header:
    st.title('Predictive Maintenance')
    st.markdown('**Sooner or Later, all machines run to a failure!**')
    st.markdown('Predictive Maintenance is a type of condition based maintenance where maintenance is only scheduled when specific conditions are met and before the equipment breaks down.')
#Data acquisition
with dataset:
    st.header('Dataset')
    uploaded_file = st.file_uploader("First of all, choose a CSV file:")
    df = uploaded_file
    st.write('Below you will find the first 5 rows of the dataset:', df.head(5))
    st.write('Below you will find the columns of the dataset:', df.columns)
    st.write('Dataset overview (lines,columns): ', df.shape)
with data_overview:

    #Data Cleaning
    st.write('The dataset contains ', klib.missingval_plot(df), 'missing values')#It shows if there are any missing values
    df_cleaned = klib.data_cleaning(df)#drops empty and sigle valued columns as well as empty and duplicate rows.
    st.write('The dataset without empty and sigle valued columns as well as empty and duplicate rows:', df_cleaned)

    #Remove outliers and numbers which does not make sense
    df=df[df["Torque [Nm]"]!=8.8]
    df=df[df["Torque [Nm]"]!=9.7]
    df=df[df["Torque [Nm]"]!=9.3]
    df=df[df["Torque [Nm]"]!=9.8]

    #Change the temperature from Klevin to Celcius
    df["Air temperature [K]"]= df["Air temperature [K]"].apply (lambda x:x-273.15)
    df["Process temperature [K]"]= df["Process temperature [K]"].apply (lambda x:x-273.15)
    df = df.rename(columns={"Air temperature [K]": "Airtem[c]", "Process temperature [K]": "Protem[c]","Product ID":"ID","Machine failure":"Machinefailure"})

    #remove duplcates
    st.write('The dataset contains ', df.drop_duplicates(inplace= True), 'duplicates. ', 'The dataset without duplicates is presented above: ')
    st.write(df)
    del df["UDI"]
    st.write('The dataset without useless columns:', df)

with data_overview:
    st.header('EDA')
    f = pd.read(df)

    st.markdown('Basic characteristics of the data:')
    st.write(df.describe())

    #Display all correlations data
    st.set_option('deprecation.showPyplotGlobalUse', False)
    klib.corr_plot(df, annot=False)
    st.write('Below you will find a Feature-correlation plot:')
    st.pyplot()
    st.write('Below you will find a Feature-correlation plot with a target variable interest (Machine failure):')
    klib.corr_plot(df,target="Machinefailure")
    st.pyplot()
    #Corelation heatmap
    st.write('Below you will find a correlation heatmap for Machine failure and possible causes: ')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    print(sns.heatmap(df[["Airtem[c]","Protem[c]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machinefailure"]].corr()))
    st.pyplot()
    #Distibution plot
    st.write('Below you will find a distribution plot for tool wear:')
    klib.dist_plot(df)
    st.pyplot()
    #Categorial plt
    st.write('Below you will find a categorial data plot:')
    klib.cat_plot(df, top=4, bottom=4)
    st.pyplot()

    #Box plots
    st.write('Below you will find a box plots for possible failure causes:')
    fig, axs = plt.subplots(1, 1)
    print(df.boxplot("Protem[c]"))
    st.pyplot()
    print(df.boxplot("Rotational speed [rpm]"))
    st.pyplot()
    print(df.boxplot("Airtem[c]"))
    st.pyplot()
    print(df.boxplot("Torque [Nm]"))
    st.pyplot()
    print(df.boxplot("Tool wear [min]"))
    st.pyplot()

    st.write('Below you will find a box plots for Process and Air Temperature:')
    print(df.boxplot(column =["Protem[c]","Airtem[c]"]))
    st.pyplot()
    st.write('Below you will find histograms for data columns:')
    print(df.hist())
    st.pyplot()

    #scatter plot grid
    st.write('Below you will find scatterplots for data every possible column pair:')
    print(df.columns)
    selections = ['Type', 'Airtem[c]', 'Protem[c]', 'Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]', 'Machinefailure']
    df5 = df[selections]
    g=sns.PairGrid(df5)
    g.map(plt.scatter)
    st.pyplot()

    #pivot
    st.write('The tables below are pivot tables for data categories: ')
    st.write(pd.pivot_table(df, index = 'Type', values = 'Protem[c]'))
    st.write(pd.pivot_table(df, index = 'Type', values = 'Tool wear [min]'))
    st.write(pd.pivot_table(df, index = ['Type','Protem[c]'], values = 'Tool wear [min]').sort_values('Tool wear [min]',ascending = False))
    st.write(pd.pivot_table(df, index = ['Type','Machinefailure'], values = 'Tool wear [min]').sort_values('Tool wear [min]',ascending = False))
    st.write(pd.pivot_table(df, index = ['Type','Machinefailure'], values = 'Protem[c]').sort_values('Protem[c]',ascending = False))
    st.write(pd.pivot_table(df, index = ['Type','Machinefailure'], values = 'Rotational speed [rpm]').sort_values('Rotational speed [rpm]',ascending = False))
    df_pivots = df[['Type', 'Airtem[c]', 'Protem[c]', 'Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]', 'Machinefailure', 'TWF', 'HDF','PWF', 'OSF', 'RNF']]
    st.write(df_pivots)

with data_modeling:
    st.header('Data Modeling')
    #Random forest
    clf = RandomForestClassifier()
    print(df.columns)
    df = df.rename(columns={"Tool wear [min]":"Toolwearmin"})
    df_model=df[['Machinefailure','Toolwearmin','Type', 'Airtem[c]', 'Protem[c]', 'Rotational speed [rpm]','Torque [Nm]']]
    #In order to take more dummies
    df_dum = pd.get_dummies(df_model)

    # We set which is our Y - What we want to predict
    X = df_dum.drop("Machinefailure", axis =1)
    y = df_dum.Machinefailure.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    st.write('X train shape (lines,columns): ', X_train.shape, 'Y train shape (lines,columns): ', y_train.shape)
    st.write('X test shape (lines,columns): ', X_test.shape, 'Y test shape (lines,columns): ', y_test.shape)

    #Rebuild the Random Forest Model
    clf.fit(X_train, y_train)
    st.write('Below you will find the predicted values for Machine Failure: ')
    st.write(clf.predict(X_test))#Its our prediction
    st.write('Below you will find the actual values for Machine Failure: ')
    st.write(y_test)#Y are the actual

    #Model Performance
    st.markdown('Model Performance')
    st.write('Accuracy: ', clf.score(X_test, y_test))
