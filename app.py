import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_path = 'data/processed/best_model.pkl'
model = joblib.load(model_path)

# Define department options
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

# Add custom CSS to adjust background colors, margins, and header text color
st.markdown("""
<style>
body {
    background-color: #000000; /* Black background for the entire app */
    color: #ffffff; /* White text for better contrast */
    margin: 0;
    padding: 0;
}
header {
    margin: 0;
    padding: 0;
}
h2, h3 {
    margin: 0;
    padding: 0;
}
.sidebar .sidebar-content {
    background-color: #1e1e1e; /* Less dark background for the sidebar */
    color: #ffffff;
}
.sidebar .sidebar-header {
    color: #00ff00 !important; /* Green color for the sidebar header text */
    margin-bottom: 20px !important; /* Space below the sidebar header */
}
.stSlider .stSlider .st-bd {
    background-color: #333333; /* Darker background for sliders */
}
.stMarkdown, .stText {
    background-color: #1e1e1e; /* Less dark background for markdown/text areas */
}
.stTextInput, .stNumberInput, .stSelectbox, .stMultiselect, .stRadio, .stCheckbox {
    background-color: #333333; /* Darker background for input fields */
    color: #ffffff; /* White text in input fields */
}
</style>
""", unsafe_allow_html=True)

# Add the prediction icon before the title
st.markdown("<h2 style='text-align: center; color: red; margin-bottom: 10px;'>📈 Employee Performance Rating</h2>", unsafe_allow_html=True)

st.markdown("""
This application predicts the performance rating of employees based on important human resources factors using a machine learning model. Complete the details on the left pane, and the app will provide you with a predicted performance rating and its probability.
""")

# Define user input fields
st.sidebar.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #00ff00;">📝 Slide to Enter Employee Details:</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar sliders and inputs
emp_department = st.sidebar.selectbox(
    'Department', 
    options=list(department_options.keys()), 
    index=0  # Default to 'Select a Department'
)
emp_department_value = department_options[emp_department]

emp_environment_satisfaction = st.sidebar.slider('Environment Satisfaction', 0, 5, 0)
emp_last_salary_hike_percent = st.sidebar.slider('Last Salary Hike Percent', 0, 30, 0)
emp_work_life_balance = st.sidebar.slider('Work-Life Balance', 0, 5, 0)
experience_years_at_company = st.sidebar.slider('Years at This Company', 0, 40, 5)
experience_years_in_role = st.sidebar.slider('Years in Current Role', 0, 40, 5)
years_since_last_promotion = st.sidebar.slider('Years Since Last Promotion', 0, 20, 0)
years_with_current_manager = st.sidebar.slider('Years with Current Manager', 0, 40, 0)

# Handle case where no department is selected
if emp_department_value is None:
    st.write("Please select a department.")
else:
    # Create DataFrame from user input
    user_data = {
        'EmpDepartment': [emp_department_value],
        'EmpEnvironmentSatisfaction': [emp_environment_satisfaction],
        'EmpLastSalaryHikePercent': [emp_last_salary_hike_percent],
        'EmpWorkLifeBalance': [emp_work_life_balance],
        'ExperienceYearsAtThisCompany': [experience_years_at_company],
        'ExperienceYearsInCurrentRole': [experience_years_in_role],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_current_manager]
    }

    # Convert to DataFrame
    user_df = pd.DataFrame(user_data)

    # Extract feature names from the model if available
    if hasattr(model, 'feature_names_in_'):
        feature_names_from_model = model.feature_names_in_
    else:
        feature_names_from_model = [
            'EmpDepartment', 'EmpEnvironmentSatisfaction', 'EmpLastSalaryHikePercent',
            'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
            'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager'
        ]

    # Check if all required features are present
    missing_features = set(feature_names_from_model) - set(user_df.columns)
    if missing_features:
        st.error(f"Missing features: {', '.join(missing_features)}")
        st.stop()

    # Ensure the DataFrame columns match the model's expected feature names
    input_data = user_df[feature_names_from_model]

    # Get predictions and probabilities
    try:
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None

        # Extract predicted class and its probability
        predicted_class = predictions[0]
        predicted_class_probability = None

        if predicted_class in model.classes_:
            predicted_class_index = model.classes_.tolist().index(predicted_class)
            predicted_class_probability = probabilities[0][predicted_class_index] * 100 if probabilities is not None else None

        # Display the results
        st.markdown("<h3 style='text-align: center; color: blue; margin-bottom: 10px; margin-top: 10px;'>Key Assessment Areas & Predicted Rating</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 2])

        with col1:
            st.markdown("<h3 style='margin-bottom: 10px;'>Assessment Data</h3>", unsafe_allow_html=True)
            st.write(user_df.T, use_container_width=True)  # Display transposed DataFrame

        with col2:
            st.markdown("<h3 style='margin-bottom: 10px;'>Performance Rating</h3>", unsafe_allow_html=True)
            st.write(f"**Predicted Performance Rating:** {predicted_class}")
            if predicted_class_probability is not None:
                st.write(f"**Estimated Probability:** {predicted_class_probability:.2f}% 🎯")

            st.markdown("<hr>", unsafe_allow_html=True)

            # Add the important icon before the note
            st.markdown("""
            **⚠️ Note:** <br>The performance rating is predicted based on the model trained with historical employee data. Ensure that the input data is accurate for reliable predictions.
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Hide Streamlit style
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
