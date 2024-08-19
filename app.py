import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessor pipeline
try:
    model = joblib.load('data/processed/best_model.pkl')
    preprocessor = joblib.load('data/processed/preprocessor.pkl')
except Exception as e:
    st.error(f"Failed to load model or preprocessor: {e}")
    st.stop()

# Define the top features used during training
top_features = [
    'YearsSinceLastPromotion', 
    'EmpEnvironmentSatisfaction',
    'EmpWorkLifeBalance', 
    'EmpLastSalaryHikePercent',
    'EmpDepartment'
]

# Department mapping
department_options = {
    'Select a Department': None,
    'Sales': 0,
    'Human Resources': 1,
    'Development': 2,
    'Data Science': 3,
    'Research & Development': 4,
    'Finance': 5
}

# Streamlit app setup
st.set_page_config(page_title="Employee Performance Rating", page_icon="📈")
st.title("Predicting Employee Performance Rating")

st.markdown("""
This application predicts the performance rating of employees based on various factors. Enter the details below, and the app will provide you with a predicted performance rating and its probability.
""")

# Define user input fields
st.sidebar.header("Enter Employee Details")
years_since_last_promotion = st.sidebar.slider('Years Since Last Promotion', 0, 20, 10)
emp_environment_satisfaction = st.sidebar.slider('Environment Satisfaction', 1.0, 5.0, 3.0)
emp_work_life_balance = st.sidebar.slider('Work-Life Balance', 1.0, 5.0, 3.0)
emp_last_salary_hike_percent = st.sidebar.slider('Last Salary Hike Percent', 1, 100, 20)

# Department selection with default to None
emp_department = st.sidebar.selectbox(
    'Department', 
    options=list(department_options.keys()), 
    index=0  # Default to 'Select a Department'
)
emp_department_value = department_options[emp_department]

# Handle case where no department is selected
if emp_department_value is None:
    st.write("Please select a department.")
else:
    # Create DataFrame from user input
    user_data = {
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'EmpEnvironmentSatisfaction': [emp_environment_satisfaction],
        'EmpWorkLifeBalance': [emp_work_life_balance],
        'EmpLastSalaryHikePercent': [emp_last_salary_hike_percent],
        'EmpDepartment': [emp_department_value],
    }

    user_df = pd.DataFrame(user_data, columns=top_features)

    st.write("User Data:")
    st.write(user_df)

    try:
        # Transform user data using the preprocessor
        user_data_transformed = preprocessor.transform(user_df)
        st.write("Transformed User Data:")
        st.write(user_data_transformed)

        # Make predictions
        predictions = model.predict(user_data_transformed)
        probabilities = model.predict_proba(user_data_transformed) if hasattr(model, 'predict_proba') else None

        # Extract predicted class and its probability
        predicted_class = predictions[0]
        predicted_class_probability = None

        # If probabilities exist, get the probability of the predicted class
        if probabilities is not None:
            predicted_class_index = model.classes_.tolist().index(predicted_class)
            predicted_class_probability = probabilities[0][predicted_class_index] * 100

        # Display results
        st.subheader("Prediction Results")
        if predicted_class_probability is not None:
            st.write(f"**Predicted Performance Rating:** {predicted_class}")
            st.write(f"**Estimated Probability:** {predicted_class_probability:.2f}%")
            st.write(f"**These results are likely to be {predicted_class_probability:.2f}% accurate.**")
        else:
            st.write(f"**Predicted Performance Rating:** {predicted_class}")

    except AttributeError as e:
        st.error(f"AttributeError: {e}. Check if the model and preprocessor versions are compatible.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Additional information
    st.markdown("""
    **Note:** The performance rating is predicted based on the model trained with historical employee data. Ensure that the input data is accurate for reliable predictions.
    """)

    # Hide Streamlit style
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
