import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
data = pd.read_csv('secondary_data.csv', sep=';')

# Load the trained models
with open('Mashroom_svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('Mashroom_knn.pkl', 'rb') as file:
    knn_model = pickle.load(file)
    
# Drop columns from the data
data = data.drop(['stem-root', 'veil-type', 'veil-color', 'spore-print-color'], axis=1)
data = data.dropna()

# Encode categorical variables
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Split the data into features (X) and target variable (y)
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Sidebar
st.sidebar.title("Options")
show_dataframe = st.sidebar.checkbox("Show Dataframe")
show_graphs = st.sidebar.checkbox("Show Graphs")
ml_model = st.sidebar.selectbox("Select ML Model", ["SVM", "KNN"])
evaluation_metrics = st.sidebar.multiselect("Select Evaluation Metrics", ["Accuracy", "Confusion Matrix", "Classification Report"])

# Main content
st.title("Mushroom Classification Dashboard")

# Show dataframe if selected
if show_dataframe:
    st.subheader("Mushroom Dataframe")
    st.dataframe(data)

# Show graphs if selected
if show_graphs:
    st.subheader("Data Visualizations")

    # Scatter Plot
    fig_scatter = px.scatter(data, x='cap-diameter', y='stem-height', color='class', title='Scatter Plot: Cap Diameter vs Stem Height', trendline='ols')
    st.plotly_chart(fig_scatter)

    # Histogram
    fig_histogram = px.histogram(data, x='cap-diameter', title='Histogram: Cap Diameter')
    st.plotly_chart(fig_histogram)

    # 3D Scatter Plot
    fig_3d = px.scatter_3d(data, x='cap-diameter', y='stem-height', z='cap-color', color='class',
                           title='3D Scatter Plot: Cap Diameter vs Stem Height vs Cap Color')
    st.plotly_chart(fig_3d)

# Model Selection and Evaluation
if ml_model == "SVM":
    selected_model = svm_model
else:
    selected_model = knn_model

st.subheader(f"{ml_model} Metrics:")

# Model Predictions
predictions = selected_model.predict(X_test)

# Evaluate selected metrics
if "Accuracy" in evaluation_metrics:
    st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")

if "Confusion Matrix" in evaluation_metrics:
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, predictions))
    # Create a confusion matrix heatmap
    cm = confusion_matrix(y_test, predictions)
    fig_confusion_matrix = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"),
                                     x=['Edible', 'Poisonous'], y=['Edible', 'Poisonous'],
                                     color_continuous_scale='blues', origin='lower')
    
    st.plotly_chart(fig_confusion_matrix)

if "Classification Report" in evaluation_metrics:
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))
    
    # Generate classification report
    report_dict = classification_report(y_test, predictions, output_dict=True)
    
    # Display classification report in DataFrame
    df_classification_report = pd.DataFrame(report_dict).transpose()
    st.dataframe(df_classification_report)