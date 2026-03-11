import streamlit as st
import joblib
import pandas as pd
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Rating",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #000000;
    color: white;
}

h1,h2,h3,h4 {
    color: white;
}

.block-container {
    padding-top: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #1e1e1e;
}

.metric-card {
    background-color:#1e1e1e;
    padding:20px;
    border-radius:10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model_path = "data/processed/best_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Ensure `best_model.pkl` exists.")
    st.stop()

model = joblib.load(model_path)

# ---------------------------------------------------
# DEPARTMENT OPTIONS
# ---------------------------------------------------
department_options = {
    "Select a Department": None,
    "Sales": 0,
    "Human Resources": 1,
    "Development": 2,
    "Data Science": 3,
    "Research & Development": 4,
    "Finance": 5
}

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#ff4b4b;'>📈 Employee Performance Rating</h1>",
    unsafe_allow_html=True
)

st.markdown(
"""
This application predicts the **performance rating of employees**
based on key **HR factors** using a machine learning model.

Adjust the employee details on the **left sidebar** to generate a prediction.
"""
)

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.markdown(
"""
<h3 style='color:#00ff00;'>📝 Enter Employee Details</h3>
""",
unsafe_allow_html=True
)

emp_department = st.sidebar.selectbox(
    "Department",
    options=list(department_options.keys()),
    index=0
)

emp_department_value = department_options[emp_department]

emp_environment_satisfaction = st.sidebar.slider(
    "Environment Satisfaction", 0, 5, 0
)

emp_last_salary_hike_percent = st.sidebar.slider(
    "Last Salary Hike Percent", 0, 30, 0
)

emp_work_life_balance = st.sidebar.slider(
    "Work-Life Balance", 0, 5, 0
)

experience_years_at_company = st.sidebar.slider(
    "Years at This Company", 0, 40, 5
)

experience_years_in_role = st.sidebar.slider(
    "Years in Current Role", 0, 40, 5
)

years_since_last_promotion = st.sidebar.slider(
    "Years Since Last Promotion", 0, 20, 0
)

years_with_current_manager = st.sidebar.slider(
    "Years with Current Manager", 0, 40, 0
)

# ---------------------------------------------------
# VALIDATE DEPARTMENT
# ---------------------------------------------------
if emp_department_value is None:
    st.warning("Please select a department from the sidebar.")
    st.stop()

# ---------------------------------------------------
# CREATE INPUT DATA
# ---------------------------------------------------
user_data = {
    "EmpDepartment": [emp_department_value],
    "EmpEnvironmentSatisfaction": [emp_environment_satisfaction],
    "EmpLastSalaryHikePercent": [emp_last_salary_hike_percent],
    "EmpWorkLifeBalance": [emp_work_life_balance],
    "ExperienceYearsAtThisCompany": [experience_years_at_company],
    "ExperienceYearsInCurrentRole": [experience_years_in_role],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "YearsWithCurrManager": [years_with_current_manager]
}

user_df = pd.DataFrame(user_data)

# ---------------------------------------------------
# FEATURE ORDER FROM MODEL
# ---------------------------------------------------
if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    feature_names = list(user_data.keys())

input_data = user_df[feature_names]

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
try:

    prediction = model.predict(input_data)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(prediction)
        probability = probs[class_index] * 100

    st.markdown("---")

    st.markdown(
        "<h2 style='text-align:center;color:#4aa3ff;'>Prediction Results</h2>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # ASSESSMENT DATA
    # ---------------------------------------------------
    with col1:

        st.markdown("### Assessment Data")

        st.dataframe(
            user_df.T.style.set_properties(**{
                "background-color": "#1e1e1e",
                "color": "white"
            }),
            use_container_width=True
        )

    # ---------------------------------------------------
    # PREDICTION RESULT
    # ---------------------------------------------------
    with col2:

        st.markdown("### Performance Rating")

        st.metric(
            label="Predicted Rating",
            value=prediction
        )

        if probability is not None:

            st.metric(
                label="Prediction Confidence",
                value=f"{probability:.2f}%"
            )

            st.progress(probability / 100)

        st.markdown("---")

        st.info(
            "⚠️ Prediction is based on historical employee performance data."
        )

except Exception as e:

    st.error(f"Prediction failed: {str(e)}")

# ---------------------------------------------------
# HIDE STREAMLIT FOOTER
# ---------------------------------------------------
st.markdown(
"""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
unsafe_allow_html=True
)
