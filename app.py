import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Function to plot precision-recall curves
def plot_precision_recall_curves(y_true, y_pred_proba, title='Precision-Recall Curves'):
    for i in range(y_true.shape[1]):
        precision, recall, _ = precision_recall_curve(y_true.iloc[:, i], y_pred_proba[:, i])
        ap_score = average_precision_score(y_true.iloc[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i} AP={ap_score:0.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower right")

# Load your dataset
df = pd.read_csv('data/cleaned_data.csv')

# Rename column names as per the provided dictionary
new_columns = {
    'Entity': 'country',
    'Year': 'year',
    'Access to electricity (% of population)': 'electricity_access_percent',
    'Access to clean fuels for cooking': 'clean_cooking_access_percent',
    'Renewable energy share in the total final energy consumption (%)': 'renewable_energy_share_percent',
    'Electricity from fossil fuels (TWh)': 'electricity_from_fossil_fuels_twh',
    'Electricity from nuclear (TWh)': 'electricity_from_nuclear_twh',
    'Electricity from renewables (TWh)': 'electricity_from_renewables_twh',
    'Low-carbon electricity (% electricity)': 'low_carbon_electricity_percent',
    'Primary energy consumption per capita (kWh/person)': 'primary_energy_consumption_per_capita_kwh_per_person',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': 'energy_intensity_mj_per_usd_gdp',
    'Value_co2_emissions_kt_by_country': 'co2_emissions_kt',
    'gdp_growth': 'gdp_growth',
    'gdp_per_capita': 'gdp_per_capita'
}

df.rename(columns=new_columns, inplace=True)

# Creating the target variable for classification
conditions = [
    (df['renewable_energy_share_percent'] <= 20),
    (df['renewable_energy_share_percent'] <= 40),
    (df['renewable_energy_share_percent'] <= 60),
    (df['renewable_energy_share_percent'] <= 80),
    (df['renewable_energy_share_percent'] <= 100)
]
choices = ['very low', 'low', 'moderate', 'high', 'very high']

df['Energy Access Classification'] = np.select(conditions, choices, default=np.nan)

# Feature and target variables
features = [
    'year',
    'country',
    'electricity_access_percent',
    'clean_cooking_access_percent',
    'renewable_energy_share_percent',
    'electricity_from_fossil_fuels_twh',
    'electricity_from_nuclear_twh',
    'electricity_from_renewables_twh',
    'low_carbon_electricity_percent',
    'primary_energy_consumption_per_capita_kwh_per_person',
    'energy_intensity_mj_per_usd_gdp',
    'co2_emissions_kt',
    'gdp_growth',
    'gdp_per_capita'
]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Energy Access Classification'], test_size=0.3, random_state=42)

# Define transformers for preprocessing
numeric_features = [
    'year',
    'electricity_access_percent',
    'clean_cooking_access_percent',
    'renewable_energy_share_percent',
    'electricity_from_fossil_fuels_twh',
    'electricity_from_nuclear_twh',
    'electricity_from_renewables_twh',
    'low_carbon_electricity_percent',
    'primary_energy_consumption_per_capita_kwh_per_person',
    'energy_intensity_mj_per_usd_gdp',
    'co2_emissions_kt',
    'gdp_growth',
    'gdp_per_capita'
]

categorical_features = ['country']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Using Support Vector Machine (SVM) model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))
])

model.fit(X_train, y_train)

# Streamlit app
st.title('Energy Access Classification App')

# Sidebar for user input
user_input = {}

# Add features to user input
for feature in features:
    if feature == 'year':
        user_input[feature] = st.sidebar.selectbox(f'Select {feature}:', df['year'].unique())
    elif feature == 'country':
        # Use st.selectbox for the 'country' feature
        user_input[feature] = st.sidebar.selectbox(f'Select {feature}:', df['country'].unique())
    else:
        user_input[feature] = st.sidebar.number_input(f'Enter {feature}:', value=0.0)

user_input_df = pd.DataFrame(user_input, index=[0])

# Check if input values are not empty
if user_input_df.isnull().values.any() or (user_input_df.astype(str) == '').any().any():
    st.warning('Please enter valid values for all input features.')
else:
    # Make a prediction based on user input
    prediction = model.predict(user_input_df)[0]

    # Display user input values
    st.subheader('User Input Values:')
    st.write(user_input_df)

    # Display the prediction
    st.subheader('Prediction:')
    st.write(f'The predicted classification for user input is {prediction}.')

    # Display classification report
    st.subheader('Classification Report:')
    y_test_pred_proba = model.predict_proba(X_test)
    report = classification_report(y_test, model.predict(X_test), target_names=choices)
    st.text(report)

    # Display confusion matrix
    st.subheader('Confusion Matrix:')
    cm = confusion_matrix(y_test, model.predict(X_test))
    st.write(cm)

    # Display precision-recall curves
    st.subheader('Precision-Recall Curves:')
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_precision_recall_curves(pd.get_dummies(y_test), y_test_pred_proba, title='Precision-Recall Curves')
    st.pyplot(fig)
