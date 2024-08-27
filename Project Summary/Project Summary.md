### PROJECT SUMMARY

- **Algorithm and Training Methods**: The project utilized the following models:

  - **Random Forest**: Chosen for its high accuracy of 97% and robustness in handling complex datasets.

  - **XGBoost**: Selected for its performance and ability to handle various types of data effectively.

  - **Logistic Regression**: Used for its simplicity and interpretability.

  - **Support Vector Machine (SVM)**: Included for its ability to find optimal decision boundaries.

  - **K-Nearest Neighbors (KNN)**: Added to assess its performance in non-linear decision boundaries.

  - **Decision Trees**: Employed to understand basic decision rules and feature importance.

  - **Naive Bayes**: Used to provide a probabilistic perspective on predictions.

  These models were chosen to ensure a comprehensive evaluation of the data with varying complexities and approaches. Cross-validation and hyperparameter tuning (using `GridSearchCV`) were employed to optimize their performance.

- **Feature Selection**: Key features were selected based on their correlation scores (≥0.1), which included:

  - **EmpDepartment**

  - **EmpEnvironmentSatisfaction**

  - **EmpLastSalaryHikePercent**

  - **EmpWorkLifeBalance**

  - **ExperienceYearsAtThisCompany**

  - **ExperienceYearsInCurrentRole**

  - **YearsSinceLastPromotion**

  - **YearsWithCurrManager**

  SHAP values confirmed the importance of these features. No dimensionality reduction techniques like PCA were used due to their negligible impact on performance compared to correlation-based selection.

- **Other Techniques and Tools**: 

  - **SMOTE** was used to address class imbalance, improving model performance.

  - **Visualization** tools like Matplotlib and Seaborn were utilized for exploratory data analysis and interpretation.

  - **Scaling/Standardization** was applied to improve model performance and consistency.

### FEATURES SELECTION / ENGINEERING

- **Most Important Features**: The following features were identified as most critical:

  - **EmpLastSalaryHikePercent**

  - **EmpEnvironmentSatisfaction**

  - **YearsSinceLastPromotion**

  - **EmpDepartment**

  These features were selected due to their high impact on model predictions, as confirmed by SHAP values.

- **Feature Transformations**: 

  - **Scaling and Standardization**: Applied to ensure that the models performed optimally.

  - **Handling Imbalance**: SMOTE was used to generate synthetic samples for minority classes, addressing class imbalance issues.

- **Feature Correlation and Interactions**: 

  - **Correlation Analysis**: Identified important features based on correlation scores of ≥0.1.

  - **Interactions**: Considered using SHAP interaction values to understand how features influenced predictions together.

### RESULTS, ANALYSIS, AND INSIGHTS

- **Interesting Relationships**: The analysis uncovered significant relationships between employee satisfaction metrics and performance ratings. Features such as `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction` had a strong influence on performance outcomes. 

- **Most Important Technique**: The implementation of SHAP for model interpretability was crucial. It provided clear insights into how each feature contributed to the predictions and guided actionable business decisions.

- **Business Problem Solutions**: 

  - **Employee Performance**: Identified key drivers of performance ratings, such as `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction`. These insights guide INX in prioritizing and monitoring these aspects to maintain and improve employee performance. Any upward adjustment on any of these three resulted in a significant increase in performance rating in the model!

  - **Department Influence**: The analysis also highlighted the importance of the department in influencing performance ratings, suggesting targeted interventions based on departmental data.

- **Additional Business Insights**: 

  - **Feature Importance**: Emphasized the need for regular monitoring of features like `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction` to ensure optimal employee performance.

  - **Scaling and Balancing**: Demonstrated the significant impact of addressing class imbalance and standardizing features on model accuracy and performance.

- **Recommendation(s)**: 
  - **Model Use**: This model can be used to predict employee performance rating with a high accuracy.
  - **Model Improvement**: The model could improve from obtaining training datasets that include score 1 or data on employee who ever rated poorly.
