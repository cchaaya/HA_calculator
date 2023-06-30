# import libariries
import streamlit as st
import time
import numpy as np
import pandas as pd
import altair as alt
import graphviz as graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
from streamlit import pyplot as st_pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import base64

# Main page
st.set_page_config(layout="wide")         
st.title("Breast Cancer")

# Add a sidebar
st.sidebar.title("Selector")
        
app_mode = st.sidebar.selectbox('Select Page',['Overview','Diagnostic Analysis','Prediction']) #three pages
if app_mode=='Overview':    
    data=pd.read_csv('https://raw.githubusercontent.com/cchaaya/HA_calculator/main/BrcadatasetFinal2.csv')

    # Display an image
    image = Image.open("R.jpeg")

    # Display image and metrics side by side in the same column
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Health Analytics")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Insights from LT")
        st.metric(label="BC in 2020", value="2,262,419  (~11.7%)")
        st.metric(label="Mortality Rate", value="6.9%")
        st.metric(label="BC-Lebanon in 2020 ", value="1,954 (~ 16.9%)")
        st.metric(label="Positive Margin Range", value=" 20% - 70%")
        st.metric(label="Cost of Re-Operation", value="up to $7,000")
        
    st.subheader("Dataset: ")
    st.markdown('13 out of 146 variables were selected based on SME')    
    st.write(data.head())

    # Introduction
    st.subheader("Introduction: BC Treatments & Positive Margin")
    st.markdown("The treatment of breast cancer includes various modalities, and the selection of the most suitable approach is determined by " 
            "evaluating multiple factors related to the patient and the disease. These considerations are thoroughly examined and discussed "
            "by a team of multidisciplinary specialists in conjunction with the patient.") 
    st.markdown("Among the treatment options available, breast surgery is a frequently employed method for cancer removal. This can be accomplished " 
            "through either a mastectomy or a lumpectomy (also known as partial mastectomy, or breast-conserving surgery (BCS)). "
             "The latter procedure involves the removal of solely the cancerous cells while preserving the natural shape of the breast, resulting in a more favorable cosmetic outcome.")

    st.subheader("Descriptive Analysis")

    # EDA - Descriptive 
    col1, col2 = st.columns([1, 1])
    with col1:    
        # Define the data
        sample_size = 321
        total_patients = 750

        # Calculate the remaining patients
        remaining_patients = total_patients - sample_size

        # Create the pie chart for Patients Distribution
        fig0 = go.Figure(data=[go.Pie(labels=['Sample Patients', 'Remaining Patients'],
                                    values=[sample_size, remaining_patients],
                                    hole=0.5)])
        fig0.update_layout(title_text="Patients Distribution", showlegend=True)
        fig0.update_traces(textinfo='value')
        col1.plotly_chart(fig0)

    with col2:    
        positive_margin_percentage = (data['Margins'] == 1).mean() * 100
        negative_margin_percentage = (data['Margins'] == 0).mean() * 100

        # Create the donut chart for Margins
        fig0 = go.Figure(data=[go.Pie(labels=['Positive Margin', 'Negative Margin'],
                                    values=[positive_margin_percentage, negative_margin_percentage],
                                    hole=0.5)])
        fig0.update_layout(title_text="Margin Distribution in our sample", showlegend=True)
        col2.plotly_chart(fig0)

    # Display the data of patients
    col1, col2 = st.columns([1, 1])
    with col1:
        # Calculate the age distribution
        age_counts = data['Age50'].value_counts()
        # Create a Pie chart for age distribution
        figa = go.Figure(data=[go.Pie(labels=['>50', '<50'], values=age_counts.values, hole=0.5)])
        figa.update_layout(title_text="Age Distribution", showlegend=False)
        figa.update_traces(textinfo='label+percent')
        col1.plotly_chart(figa, use_container_width=True)

    with col2:
        # Calculate the BMI distribution
        bmi_counts = data['BMI'].value_counts().reset_index()
        # Group BMI values into two categories: above 30 and below 30
        bmi_above_30 = bmi_counts[data['BMI'] > 30]
        bmi_below_30 = bmi_counts[data['BMI'] <= 30]
        # Create a Pie chart for BMI above 30
        fig_above_30 = go.Figure(data=[go.Pie(labels=['Above 30', 'Below 30'], values=[bmi_above_30['BMI'].sum(), bmi_below_30['BMI'].sum()], hole=0.5)])
        fig_above_30.update_layout(title_text="BMI Distribution", showlegend=False)
        fig_above_30.update_traces(textinfo='label+percent')
        st.plotly_chart(fig_above_30, use_container_width=True)


elif app_mode == 'Diagnostic Analysis':
    # Main page2
    st.title("THE RISK FACTORS ASSOCIATED WITH POSITIVE MARGIN IN BREAST CANCER PARTIAL MASTECTOMY")
    data=pd.read_csv('https://raw.githubusercontent.com/cchaaya/HA_app/main/BrcadatasetFinal2.csv') 
    
    ## Start filter
    st.sidebar.subheader("Filter Data")
    number_of_tumors_filter = st.sidebar.multiselect("Select Number of Tumors", [1, 2, 3], key="number_of_tumors_multiselect")
    age_range = st.sidebar.slider(
        "Select age range",
        min_value=int(data['Age_at_diag'].min()),
        max_value=int(data['Age_at_diag'].max()),
        value=(int(data['Age_at_diag'].min()), int(data['Age_at_diag'].max())),
        key="age_range_slider"
    )
    tumor_size_range = st.sidebar.slider(
        "Select tumor size range",
        min_value=float(data['Tumor_size'].min()),
        max_value=float(data['Tumor_size'].max()),
        value=(float(data['Tumor_size'].min()), float(data['Tumor_size'].max())),
        key="tumor_size_range_slider"
    )
    bmi_range = st.sidebar.slider(
        "Select BMI range",
        min_value=float(data['BMI'].min()),
        max_value=float(data['BMI'].max()),
        value=(float(data['BMI'].min()), float(data['BMI'].max())),
        key="bmi_range_slider"
    )
    extent_calcification_filter = st.sidebar.checkbox("Extent_of_Calcification", value=False, key="extent_calcification_checkbox")
    wire_localization_filter = st.sidebar.checkbox("Wire_localization", value=False, key="wire_localization_checkbox")
    margin_filter = st.sidebar.selectbox("Select Margins", ["All", "Positive", "Negative"], key="margin_filter_select")

    filtered_data = data  # Start with the full dataset

    if number_of_tumors_filter:
        filtered_data = filtered_data[filtered_data['Number_of_tumors'].isin(number_of_tumors_filter)]

    if isinstance(age_range, tuple):
        filtered_data = filtered_data[(filtered_data['Age_at_diag'] >= age_range[0]) & (filtered_data['Age_at_diag'] <= age_range[1])]
    else:
        filtered_data = filtered_data[filtered_data['Age_at_diag'] == age_range]

    filtered_data = filtered_data[(filtered_data['Tumor_size'] >= tumor_size_range[0]) & (filtered_data['Tumor_size'] <= tumor_size_range[1])]
    filtered_data = filtered_data[(filtered_data['BMI'] >= bmi_range[0]) & (filtered_data['BMI'] <= bmi_range[1])]

    if extent_calcification_filter:
        filtered_data = filtered_data[filtered_data['Extent_of_Calcification'] == 1]

    if wire_localization_filter:
        filtered_data = filtered_data[filtered_data['Wire_localization'] == 1]

    if margin_filter == "Positive":
        filtered_data = filtered_data[filtered_data['Margins'] == 1]
    elif margin_filter == "Negative":
        filtered_data = filtered_data[filtered_data['Margins'] == 0]

    ## end filter

    # Display Number of tumors distribution
    tumor_counts = filtered_data['Number_of_tumors'].value_counts()
    labels = tumor_counts.index.tolist()
    sizes = tumor_counts.values.tolist()

    fig1 = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent', hole=0.4)])
    fig1.update_layout(
        title_text="Number of Tumors",
        showlegend=False,
     )

    # Display Number of Tumors vs Margins
    # Group the data by the number of tumors and margins, and count the occurrences
    grouped_data = data.groupby(['Number_of_tumors', 'Margins']).size().unstack()
    # Calculate the total count for each number of tumors
    total_counts = grouped_data.sum(axis=1)
    # Calculate the percentage of each section (positive and negative margin) for each number of tumors
    positive_margin_percentages = (grouped_data[grouped_data.columns[grouped_data.columns >= 1]].sum(axis=1) / total_counts) * 100
    negative_margin_percentages = (grouped_data[grouped_data.columns[grouped_data.columns == 0]].sum(axis=1) / total_counts) * 100
    # Create the stacked bar chart with percentages
    fig2 = go.Figure(data=[
        go.Bar(name='Negative Margins',
            x=grouped_data.index,
            y=grouped_data[0],
            marker_color='blue',
            text=negative_margin_percentages.round(1).astype(str) + '%',
            textposition='inside'),
        go.Bar(name='Positive Margins',
            x=grouped_data.index,
            y=grouped_data.sum(axis=1) - grouped_data[0],
            marker_color='red',
            text=positive_margin_percentages.round(1).astype(str) + '%',
            textposition='inside')
    ])
    fig2.update_layout(
        title='Number of Tumors vs. Margin',
        xaxis_title='Number of Tumors',
        yaxis_title='Count',
        barmode='stack'
    )
    # Display Nb of Tumors and nb of tumor vs margins plots side by side
    col1, col2 = st.columns([1, 1])
    with col1:
          st.plotly_chart(fig1)

    with col2:
         st.plotly_chart(fig2)
    
    # Display Extent of Calcification vs. Margin
    calcification_margin_counts = filtered_data.groupby(['Extent_of_Calcification', 'Margins']).size().unstack()
    fig3 = go.Figure()
    for column in calcification_margin_counts.columns:
        fig3.add_trace(go.Bar(x=calcification_margin_counts.index, y=calcification_margin_counts[column], name=str(column)))
    fig3.update_layout(
        xaxis_title='Extent of Calcification',
        yaxis_title='Count',
        title='Extent of Calcification vs. Margin'
    )

    # Display Wire Localization vs. Margin
    wire_margin_counts = filtered_data.groupby(['Wire_localization', 'Margins']).size().unstack()
    fig4 = go.Figure()
    for column in wire_margin_counts.columns:
        fig4.add_trace(go.Bar(x=wire_margin_counts.index, y=wire_margin_counts[column], name=str(column)))
    fig4.update_layout(
        xaxis_title='Wire Localization',
        yaxis_title='Count',
        title='Wire Localization vs. Margin'
    )

    # Display Extent of Calcification and Wire Localization plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig3)
    with col2:
        st.plotly_chart(fig4)
 
    # Display BMI and Age scatter plot vs Positive Margins
    margins_0 = filtered_data[filtered_data['Margins'] == 0]
    margins_1 = filtered_data[filtered_data['Margins'] == 1]
    fig5 = go.Figure()
    # Create scatter plot with Age_at_diag on x-axis and BMI on y-axis for Margins = 0
    fig5.add_trace(go.Scatter(
        x=margins_0['Age_at_diag'],
        y=margins_0['BMI'],
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='Margins = 0'
    ))
    # Create scatter plot with Age_at_diag on x-axis and BMI on y-axis for Margins = 1
    fig5.add_trace(go.Scatter(
        x=margins_1['Age_at_diag'],
        y=margins_1['BMI'],
        mode='markers',
        marker=dict(color='red', opacity=0.6),
        name='Margins = 1'
    ))
    fig5.update_layout(
        xaxis=dict(title='Age at Diagnosis'),
        yaxis=dict(title='BMI'),
        title='Age vs BMI',
        showlegend=True
    )

    # Count occurrences of each value in the "BMI" column
    bmi_counts = filtered_data['BMI'].apply(lambda x: 'Group 1 (<30)' if x < 30 else 'Group 2 (>=30)').value_counts()
    # Create the BMI pie chart
    fig6 = go.Figure(data=[go.Pie(labels=bmi_counts.index, values=bmi_counts)])
    fig6.update_layout(
        title_text="BMI Distribution",
        showlegend=True
    )

       # Display Age and BMI plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig5)
    with col2:
        st.plotly_chart(fig6)

    # Display Tumor size grouped 
    group1 = filtered_data[filtered_data['Tumor_size'] <= 1]
    group2 = filtered_data[(filtered_data['Tumor_size'] > 1) & (filtered_data['Tumor_size'] <= 5)]
    # Calculate the count of tumor sizes in each group
    group1_count = len(group1)
    group2_count = len(group2)
    # Create the donut pie chart for tumor size
    fig7 = go.Figure(data=[go.Pie(
        labels=['Group 1 (<1)', 'Group 2 (>1)'],
        values=[group1_count, group2_count],
        hole=0.4,
        marker_colors=['lightblue', 'lightgreen'],
        textinfo='percent'
    )])
    fig7.update_layout(
        title='Tumor Size Grouping',
        showlegend=True
    )
    # Display Tumor Size distribution
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Tumor_size'], mode='markers', marker=dict(symbol='circle-open', size=6)))
    fig8.add_trace(go.Scatter(x=filtered_data[filtered_data['Margins'] == 1].index,
                            y=filtered_data[filtered_data['Margins'] == 1]['Tumor_size'],
                            mode='markers',
                            marker=dict(symbol='circle', size=10, color='red'),
                            name='Positive Margins'))

    fig8.update_layout(
        title='Tumor Size',
        xaxis_title='Sample',
        yaxis_title='Tumor Size'
    )

       # Display Tumor Size and Tumor Size distribution side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig7)
    with col2:
        st.plotly_chart(fig8)

    #Page3: Build a Logistic regression Model including the significant predictors & save it
    # Specify X, y and separate the target variable
    X = data[['BMI', 'Number_of_tumors', 'Extent_of_Calcification', 'Wire_localization']]
    y = data['Margins']
    # Split the data into training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=50)
    # Create a logistic regression model
    model = LogisticRegression()
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Use the model to make predictions on the test data
    y_pred = model.predict(X_test)
    # Save the trained model to a file
    filename = 'logreg_model.sav'
    pickle.dump(model, open(filename, 'wb'))

elif app_mode == 'Prediction':
    st.title('CALCULATOR')
    st.subheader("MARGIN PREDICTION CALCULATOR FOR BREAST CANCER PARTIAL MASTECTOMY")
    # Display an image
    image2 = Image.open("medicalrec.PNG")    
    st.image(image2, use_column_width=False)   
    st.write('Sir/Mme , YOU need to fill all necessary informations in order    to get a reply to your request !')    
    st.sidebar.header("Information about the Patient :")    
    gender_dict = {"Female":1,"Male":2}    
    feature_dict = {"Yes":1,"No":2}    
    edu={'Graduate':1,'Not Graduate':2}    
    Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))
    Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))    
    Dependents=st.sidebar.radio('Dependents',options=['0','1' , '2' , '3+'])    
    Education=st.sidebar.radio('Education',tuple(edu.keys()))     
    class_0, class_3, class_1, class_2 = 0, 0, 0, 0
    if Dependents == '0':
        class_0 = 1
    elif Dependents == '1':
        class_1 = 1
    elif Dependents == '2':
        class_2 = 1
    else:
        class_3 = 1

    st.sidebar.header("Diagnostic information :") 
    Age = st.sidebar.slider('Age at diag', 0, 100, 0)
    Tumor_size = st.sidebar.slider('Tumor Size', 0.00, 5.00, format="%.2f")
    # Set the desired range for height and weight
    height = st.sidebar.slider(
        "Select Height",
        min_value=140.0,
        max_value=200.0,
        value=160.0,
        step=1.0
    )
    weight = st.sidebar.slider(
        "Select Weight",
        min_value=40.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    # Display the selected height, weight, and calculated BMI
    st.sidebar.write("Calculated BMI:", round(bmi,2))
    BMI = st.sidebar.number_input('BMI', min_value=15.00, max_value=50.00, value=bmi, step=0.01, format="%.2f")
    Number_of_tumors=st.sidebar.radio('Number of Tumors',options=[0, 1 , 2, 3])
    Extent_of_calcification=st.sidebar.radio('Extent_of_calcification',tuple(feature_dict.keys()))
    Wire_localization=st.sidebar.radio('Wire_localization',tuple(feature_dict.keys()))  
    
    @st.cache(suppress_st_warning=True)
    def get_fvalue(val):    
        feature_dict = {"No":1,"Yes":2}    
        for key,value in feature_dict.items():        
            if val == key:            
                return value
    def get_value(val,my_dict):    
        for key,value in my_dict.items():        
            if val == key:            
                return value
    data1 = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': [class_0, class_1, class_2, class_3],
        'Education': Education,
        'Age': Age,
        'Tumor_size': Tumor_size,
        'BMI': BMI,
        'Number_of_tumors': Number_of_tumors,
        'Extent_of_calcification': Extent_of_calcification,
        'Wire_localization': Wire_localization,
    }

    feature_list = [
        #data1['Age'],
        #data1['Tumor_size'],
        data1['BMI'],
        data1['Number_of_tumors'],
        get_value(data1['Extent_of_calcification'], feature_dict),
        get_value(data1['Wire_localization'], feature_dict),
        #get_value(data1['Gender'], gender_dict),
        #get_value(data1['Married'], feature_dict),
        # data1['Dependents'][0],
        # data1['Dependents'][1],
        # data1['Dependents'][2],
        # data1['Dependents'][3],
        # get_value(data1['Education'], edu),
        # data1['Property_Area'][0],
        # data1['Property_Area'][1],
        # data1['Property_Area'][2],
    ]

    single_sample = np.array(feature_list).reshape(1, -1)
  
    if st.button("Predict"):
        file_ = open("simulator.PNG", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        
        file = open("highrisk.PNG", "rb")
        contents = file.read()
        data_url_no = base64.b64encode(contents).decode("utf-8")
        file.close()
        
        loaded_model = pickle.load(open('logreg_model.sav', 'rb'))
        prediction = loaded_model.predict(single_sample)
        
        if prediction[0] == 1:
            st.error('According to our model and based on the information provided, ' 
                     'your patient is at high risk to have a positive margin')
            st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">', unsafe_allow_html=True)
        elif prediction[0] == 0:
            st.success('According to our model and based on the information provided, your patient is not at high risk to have a positive margin')
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">', unsafe_allow_html=True)

# footer = 'Prepared By "My_name"'
# st.markdown(f'<div style="position: fixed; bottom: 0;">{footer}</div>', unsafe_allow_html=True)
