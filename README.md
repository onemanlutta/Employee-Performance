# Employee Performance Rating

![image](https://github.com/user-attachments/assets/35e12817-376e-447e-aece-883b29f7c04c)



## Project Overview

### INX Future Inc Employee Performance - Project

INX Future Inc (referred to as INX) is a leading provider of data analytics and automation solutions with over 15 years of global business presence. INX has been consistently rated as one of the top 20 best employers for the past five years. Known for its employee-friendly human resource policies, which are widely considered industry best practices, INX has maintained a strong reputation.

However, in recent years, employee performance indices have shown concerning trends. This issue has escalated to the point where service delivery complaints have increased, and client satisfaction levels have dropped by 8 percentage points. 

The CEO, Mr. Brain, is aware of these issues but is hesitant to take actions that could potentially demoralize the workforce, as this could further degrade performance and damage INX's reputation as a top employer. To address these challenges, Mr. Brain, who has a background in data science, initiated this project to analyze current employee data and uncover the core causes of the performance decline. The goal is to identify clear indicators of underperformance so that any necessary corrective actions can be taken without broadly impacting employee morale.

### Project Objectives and Expected Insights

This project aims to deliver the following insights:

1. **Department-wise Performance Analysis**: Understanding the performance levels across different departments.
2. **Top 3 Important Factors Affecting Employee Performance**: Identifying the most significant factors contributing to employee performance.
3. **Performance Prediction Model**: Developing a trained model that can predict employee performance based on various factors. This model will also assist in the hiring process.
4. **Recommendations for Performance Improvement**: Providing actionable recommendations based on data analysis to improve overall employee performance.

### Project Data

The employee performance data for INX Future Inc. can be downloaded from the following link: [INX Future Inc Employee Performance Data](http://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8).

## Project Structure

The project is organized into the following directories:

```
├── Project Summary
│   ├── Requirements.md
|   ├── Project Summary.ipynb
│   ├── Project Summary.md
│   └── Analysis.md
├── data
│   ├── external           # External data sources
│   ├── processed          # Preprocessed and cleaned data
│   └── raw                # Original, unprocessed data
├── src
│   ├── Data Processing
│   │   ├── data_processing.ipynb           # Data loading and preprocessing
│   │   └── data_exploratory_analysis.ipynb  # Exploratory Data Analysis (EDA)
│   ├── models
│   │   ├── train_model.ipynb               # Model training
│   │   └── predict_model.ipynb             # Model prediction
│   └── visualization
│       └── visualize.ipynb                 # Data visualization
├── references                              # Reference documents and research papers
├── app.py                                  # Code for the Streamlit App
├── requirements.txt                        # Requirement libraries for running the Streamlit App

```
## PROJECT DATA EXPLORATION SUMMARY - Highlights

- Performance Rating
  
  ![image](https://github.com/user-attachments/assets/53bbb45e-a714-42b7-aacc-2959fe539c77)

- Performance Rating vs Department
  
![image](https://github.com/user-attachments/assets/737a333a-d8e0-4d94-ad82-11be519a181a)

- Performance Rating vs Education Background
  
![image](https://github.com/user-attachments/assets/8109ad75-98d3-46c0-839f-b90b304256d2)

- Performance Rating vs Gender
  
![image](https://github.com/user-attachments/assets/ba7106f9-708b-42be-83ab-82252d23a2c4)

- Performance Rating vs Years with Current Manager
  
![image](https://github.com/user-attachments/assets/dc9e00dc-4237-4ebe-89ff-d5a6e6f6fb7d)

- Performance Rating vs Years Since Last Promotion
  
![image](https://github.com/user-attachments/assets/9a8829c6-cffa-4426-a641-f3f954a61abf)

- Performance Rating vs Last Salary Hike Percent
  
![image](https://github.com/user-attachments/assets/9eabdabb-6bbf-446a-ba50-d01ca163874f)

- Performance Rating vs Job Level
  
![image](https://github.com/user-attachments/assets/f9317edf-3030-4735-8d18-2c45ccc1348f)


## MACHINE LEARNING PROJECT SUMMARY

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
  ![image](https://github.com/user-attachments/assets/ded96453-293b-43a1-896f-c1c5c91eae18)


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

```plain text
Logistic Regression Accuracy: 0.8881829733163914
Confusion Matrix:
 [[250  25   3]
 [ 20 200  30]
 [  0  10 249]]
Classification Report:
               precision    recall  f1-score   support

           2       0.93      0.90      0.91       278
           3       0.85      0.80      0.82       250
           4       0.88      0.96      0.92       259

    accuracy                           0.89       787
   macro avg       0.89      0.89      0.89       787
weighted avg       0.89      0.89      0.89       787

Best Parameters: {'C': 10, 'penalty': 'l1', 'solver': 'saga'}
Fitting 5 folds for each of 75 candidates, totalling 375 fits
SVC Accuracy: 0.9326556543837357
Confusion Matrix:
 [[264  13   1]
 [ 13 216  21]
 [  1   4 254]]
Classification Report:
               precision    recall  f1-score   support

           2       0.95      0.95      0.95       278
           3       0.93      0.86      0.89       250
           4       0.92      0.98      0.95       259

    accuracy                           0.93       787
   macro avg       0.93      0.93      0.93       787
weighted avg       0.93      0.93      0.93       787

Best Parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
Fitting 5 folds for each of 270 candidates, totalling 1350 fits
Decision Tree Accuracy: 0.9466327827191868
Confusion Matrix:
 [[265  13   0]
 [  7 236   7]
 [  3  12 244]]
Classification Report:
               precision    recall  f1-score   support

           2       0.96      0.95      0.96       278
           3       0.90      0.94      0.92       250
           4       0.97      0.94      0.96       259

    accuracy                           0.95       787
   macro avg       0.95      0.95      0.95       787
weighted avg       0.95      0.95      0.95       787

Best Parameters: {'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 2}
Fitting 3 folds for each of 50 candidates, totalling 150 fits
Best parameters found:  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0}
Best score found:  0.9404761904761904
XGBoost Accuracy: 0.925
Classification Report:
               precision    recall  f1-score   support

           2       0.86      0.88      0.87        49
           3       0.94      0.96      0.95       268
           4       0.91      0.74      0.82        43

    accuracy                           0.93       360
   macro avg       0.90      0.86      0.88       360
weighted avg       0.92      0.93      0.92       360

Confusion Matrix:
 [[ 43   6   0]
 [  7 258   3]
 [  0  11  32]]
Best Parameters: {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0}
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Best parameters found:  {'subsample': 0.9, 'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 3, 'learning_rate': 0.01}
Best score found:  0.9452380952380952
Gradient Boosting Accuracy: 0.9166666666666666
Confusion Matrix:
 [[ 41   8   0]
 [  9 258   1]
 [  0  12  31]]
Classification Report:
               precision    recall  f1-score   support

           2       0.82      0.84      0.83        49
           3       0.93      0.96      0.95       268
           4       0.97      0.72      0.83        43

    accuracy                           0.92       360
   macro avg       0.91      0.84      0.87       360
weighted avg       0.92      0.92      0.92       360

Fitting 5 folds for each of 96 candidates, totalling 480 fits
Best parameters found:  {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (150, 100), 'learning_rate': 'constant', 'solver': 'adam'}
Best score found:  0.9542234332425068
ANN (MLP) Accuracy: 0.9555273189326556
Confusion Matrix:
 [[271   6   1]
 [ 16 228   6]
 [  1   5 253]]
Classification Report:
               precision    recall  f1-score   support

           2       0.94      0.97      0.96       278
           3       0.95      0.91      0.93       250
           4       0.97      0.98      0.97       259

    accuracy                           0.96       787
   macro avg       0.96      0.95      0.96       787
weighted avg       0.96      0.96      0.96       787

Bagging MLP Accuracy: 0.951715374841169
Bagging MLP Confusion Matrix:
 [[273   3   2]
 [ 18 223   9]
 [  1   5 253]]
Bagging MLP Classification Report:
               precision    recall  f1-score   support

           2       0.93      0.98      0.96       278
           3       0.97      0.89      0.93       250
           4       0.96      0.98      0.97       259

    accuracy                           0.95       787
   macro avg       0.95      0.95      0.95       787
weighted avg       0.95      0.95      0.95       787

Fitting 5 folds for each of 20 candidates, totalling 100 fits
Best parameters found:  {'n_neighbors': 3, 'p': 2, 'weights': 'distance'}
Best score found:  0.9149863760217984
KNN Accuracy: 0.9110546378653113
Confusion Matrix:
 [[270   4   4]
 [ 39 194  17]
 [  4   2 253]]
Classification Report:
               precision    recall  f1-score   support

           2       0.86      0.97      0.91       278
           3       0.97      0.78      0.86       250
           4       0.92      0.98      0.95       259

    accuracy                           0.91       787
   macro avg       0.92      0.91      0.91       787
weighted avg       0.92      0.91      0.91       787

Bagging KNN Accuracy: 0.9072426937738246
Bagging KNN Confusion Matrix:
 [[269   6   3]
 [ 41 191  18]
 [  3   2 254]]
Bagging KNN Classification Report:
               precision    recall  f1-score   support

           2       0.86      0.97      0.91       278
           3       0.96      0.76      0.85       250
           4       0.92      0.98      0.95       259

    accuracy                           0.91       787
   macro avg       0.91      0.90      0.90       787
weighted avg       0.91      0.91      0.90       787

Fitting 5 folds for each of 36 candidates, totalling 180 fits
Best parameters found:  {'subsample': 0.9, 'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 3, 'learning_rate': 0.01}
Best score found:  0.9452380952380952
Random Forest Accuracy: 0.9720457433290979
Confusion Matrix:
 [[271   7   0]
 [  7 240   3]
 [  0   5 254]]
Classification Report:
               precision    recall  f1-score   support

           2       0.97      0.97      0.97       278
           3       0.95      0.96      0.96       250
           4       0.99      0.98      0.98       259

    accuracy                           0.97       787
   macro avg       0.97      0.97      0.97       787
weighted avg       0.97      0.97      0.97       787


```

![image](https://github.com/user-attachments/assets/9d4c8577-7498-4c5e-b8c8-b2dc4e9f3da3)

## Highlights for the Model Analysis

#### 1. **Logistic Regression**

**Performance:**

- **Accuracy**: 0.89

- **Best Parameters**: {'C': 10, 'penalty': 'l1', 'solver': 'saga'}

**Interpretation:**

Logistic Regression achieved an accuracy of 89%. The confusion matrix reveals the following:

  - **Class 1**: High precision and recall.

  - **Class 2**: Precision = 0.93, Recall = 0.90, F1-Score = 0.91.

  - **Class 3**: Moderate precision and recall.

  - **Class 4**: Reasonably good precision and recall.

The model's performance means it effectively identifies true positives with a relatively low rate of false positives and negatives. It handles binary outcomes well and provides clear insights into feature impacts.

**Application to Predicting Employee Performance:**

- **Merits**: Provides easy interpretability, useful for understanding which factors impact employee ratings. Good for baseline models.

- **Demerits**: May not capture complex relationships or interactions between features. Limited to linear relationships.

#### 2. **Support Vector Classifier (SVC)**

**Performance:**

- **Accuracy**: 0.93

- **Best Parameters**: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}

**Interpretation:**

SVC achieved a high accuracy of 93%. The confusion matrix indicates:

  - **Class 2**: Precision = 0.95, Recall = 0.95.

  - **Class 3**: High precision and recall.

  - **Class 4**: Effective performance, though with slightly lower F1-score compared to Class 2.

The model captures complex decision boundaries well, reflecting its ability to handle high-dimensional data and non-linear relationships.

**Application to Predicting Employee Performance:**

- **Merits**: Excellent for handling complex patterns and non-linear data. Provides strong classification performance.

- **Demerits**: Computationally intensive and requires careful hyperparameter tuning.

#### 3. **Decision Tree**

**Performance:**

- **Accuracy**: 0.95

- **Best Parameters**: {'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 2}

**Interpretation:**

Decision Tree achieved an accuracy of 95%. The confusion matrix shows:

  - **Class 4**: Highest precision and recall.

  - **Class 2**: Good performance but slightly less compared to Class 4.

The model is interpretable and reveals how different features impact predictions. However, its performance indicates effective classification with clear decision rules.

**Application to Predicting Employee Performance:**

- **Merits**: Easy to interpret and visualize decision-making processes. Good for understanding feature importance.

- **Demerits**: Prone to overfitting and may not generalize well on complex datasets.

#### 4. **XGBoost**

**Performance:**

- **Accuracy**: 0.93 (on a subset of the data)

- **Best Parameters**: {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0}

**Interpretation:**

XGBoost achieved an accuracy of 93%. The confusion matrix reveals:

  - **Class 2**: High precision and recall.

  - **Class 4**: Lower F1-score, indicating some misclassification.

XGBoost handles high-dimensional data well and is robust to overfitting, providing strong performance with complex datasets.

**Application to Predicting Employee Performance:**

- **Merits**: Handles large datasets and complex relationships efficiently. Robust and effective.

- **Demerits**: Requires tuning and can be complex to interpret.

#### 5. **Gradient Boosting**

**Performance:**

- **Accuracy**: 0.92

- **Best Parameters**: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (150, 100), 'learning_rate': 'constant', 'solver': 'adam'}

**Interpretation:**

Gradient Boosting achieved an accuracy of 92%. The confusion matrix indicates:

  - **Class 3**: High precision and recall.

  - **Other Classes**: Slightly lower precision and recall.

The model effectively captures complex patterns but requires careful parameter tuning to optimize performance.

**Application to Predicting Employee Performance:**

- **Merits**: Captures complex feature interactions and relationships. Good performance on varied datasets.

- **Demerits**: Computationally intensive and requires fine-tuning.

#### 6. **Artificial Neural Network (MLP)**

**Performance:**

- **Accuracy**: 0.96

- **Best Parameters**: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (150, 100), 'learning_rate': 'constant', 'solver': 'adam'}

**Interpretation:**

MLP achieved the highest accuracy of 96%. The confusion matrix shows:

  - **All Classes**: Very high precision and recall.

The model's deep learning approach allows it to model intricate patterns, providing robust classification.

**Application to Predicting Employee Performance:**

- **Merits**: Excellent accuracy and capability to model complex relationships. Effective for high-dimensional data.

- **Demerits**: Requires significant computational resources and can be less interpretable.

#### 7. **Bagging MLP**

**Performance:**

- **Accuracy**: 0.95

- **Best Parameters**: {'n_neighbors': 3, 'p': 2, 'weights': 'distance'}

**Interpretation:**

Bagging MLP achieved an accuracy of 95%. The confusion matrix highlights:

  - **Class 3**: Improved precision and recall.

  - **Other Classes**: Slightly improved stability and performance compared to standard MLP.

Bagging enhances the base MLP model's stability and performance, reducing variance.

**Application to Predicting Employee Performance:**

- **Merits**: Provides improved stability and performance over standard MLP. Effective for handling variations in the data.

- **Demerits**: More complex implementation compared to standard models.

#### 8. **K-Nearest Neighbors (KNN)**

**Performance:**

- **Accuracy**: 0.91

- **Best Parameters**: {'n_neighbors': 3, 'p': 2, 'weights': 'distance'}

**Interpretation:**

KNN achieved an accuracy of 91%. The confusion matrix reveals:

  - **Class 2**: Reasonable precision and recall.

  - **Other Classes**: Slightly lower performance compared to other models.

KNN's performance varies with the choice of `k` and distance metric. It is effective for smaller datasets.

**Application to Predicting Employee Performance:**

- **Merits**: Simple and intuitive. Useful for smaller datasets.

- **Demerits**: Performance can degrade with larger datasets and high dimensionality. Computationally expensive during inference.

#### 9. **Bagging KNN**

**Performance:**

- **Accuracy**: 0.91

- **Best Parameters**: {'subsample': 0.9, 'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 3, 'learning_rate': 0.01'}

**Interpretation:**

Bagging KNN achieved the same accuracy as standard KNN but offers improved stability. The confusion matrix shows:

  - **Similar performance**: With slight improvements due to bagging.

Bagging helps to enhance the performance and stability of the KNN model.

**Application to Predicting Employee Performance:**

- **Merits**: Improved stability and performance over standard KNN.

- **Demerits**: Similar limitations to KNN, with possible challenges in larger datasets and high dimensionality.

#### 10. **Random Forest**

**Performance:**

- **Accuracy**: 0.97

- **Best Parameters**: {'subsample': 0.9, 'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 3, 'learning_rate': 0.01'}

**Interpretation:**

Random Forest achieved an exceptional accuracy of 97%. The confusion matrix indicates:

  - **All Classes**: Very high precision and recall.

Random Forest's ensemble approach effectively manages various data patterns and reduces overfitting, providing highly reliable predictions.

**Application to Predicting Employee Performance:**

- **Merits**: High accuracy, robustness, and ability to handle complex interactions. Effective for large datasets.

- **Demerits**: Less interpretable than single decision trees and computationally intensive.

---

This comprehensive analysis provides an understanding of each model's performance, including their metrics and confusion matrix interpretation. Each model's merits and demerits are outlined, giving insights into their application for predicting employee performance.

- **Interesting Relationships**: The analysis uncovered significant relationships between employee satisfaction metrics and performance ratings. Features such as `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction` had a strong influence on performance outcomes.

![image](https://github.com/user-attachments/assets/e1018ca6-134b-448c-a112-1a06b7202344)


- **Most Important Technique**: The implementation of SHAP for model interpretability was crucial. It provided clear insights into how each feature contributed to the predictions and guided actionable business decisions.

![image](https://github.com/user-attachments/assets/3d0f0baf-c663-4a8f-83b3-1a66d563f3f4)

![image](https://github.com/user-attachments/assets/fd00f22c-797f-45fc-ac90-2cc33e6d1b92)

![image](https://github.com/user-attachments/assets/79b5f764-ef9b-437b-a0bf-1c01ddc71e4c)

![image](https://github.com/user-attachments/assets/522f95d7-b316-48e1-bcfb-f545f95e6446)

![image](https://github.com/user-attachments/assets/92ad4efe-c1d0-40fb-a21f-55eaf0be0d35)



- **Business Problem Solutions**: 

  - **Employee Performance**: Identified key drivers of performance ratings, such as `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction`. These insights guide INX in prioritizing and monitoring these aspects to maintain and improve employee performance. Any upward adjustment on any of these three resulted in a significant increase in performance rating in the model!

  Performance Rating 2: ![image](https://github.com/user-attachments/assets/660e0fa8-1ca0-4af7-9301-85ba8939480c)

  Performance Rating 3: ![image](https://github.com/user-attachments/assets/cde82b22-c1a4-4251-943b-571405710015)

  Performance Rating 4: ![image](https://github.com/user-attachments/assets/f0d36d73-9af6-4cd0-9247-748dd26e78ed)


  - **Department Influence**: The analysis also highlighted the importance of the department in influencing performance ratings, suggesting targeted interventions based on departmental data.

- **Additional Business Insights**: 

  - **Feature Importance**: Emphasized the need for regular monitoring of features like `EmpLastSalaryHikePercent` and `EmpEnvironmentSatisfaction` to ensure optimal employee performance.

  - **Scaling and Balancing**: Demonstrated the significant impact of addressing class imbalance and standardizing features on model accuracy and performance.

- **Recommendation(s)**: 
  - **Model Use**: This model can be used to predict employee performance rating with a high accuracy.
  ```plain text
   Your predicted Performance Rating is 2 and this has been with at least 88.00% accuracy.
   Your Predicted Performance Rating is 3 at an estimated Probability of 87.00%.
   Your Predicted Performance Rating is 2 at an estimated Probability of 47.50%.
   Your Predicted Performance Rating is 4 at an estimated Probability of 88.00%.
  ```
  - **Model Improvement**: The model could improve from obtaining training datasets that include score 1 or data on employee who ever rated poorly.



## Methodology (How to Implement / Use this Repository)

### 1. Data Loading and Preprocessing
- The dataset is loaded and inspected for missing values, outliers, and inconsistencies.
- Data is cleaned and processed, including handling categorical variables, scaling, and normalization.
- The processed data is saved in the `data/processed` directory.

### 2. Exploratory Data Analysis (EDA)
- Detailed exploratory analysis is conducted to understand data distributions and correlations between variables.
- Insights from EDA guide feature selection and engineering.

### 3. Feature Selection and Engineering
- Feature selection techniques, such as correlation analysis and feature importance from models, are used to select relevant features.
- New features are engineered based on domain knowledge and data analysis.

### 4. Model Training and Evaluation
- Multiple models, including K-Nearest Neighbors (KNN), Random Forest, and others, are trained using the selected features.
- Models are evaluated using metrics like accuracy, precision, recall, and F1-score.

### 5. Results and Insights
- The performance of each model is compared, and the best-performing model is identified.
- Insights into the key factors influencing employee performance are discussed.

### 6. Deployment
- The best model is saved for deployment.
- A simple script is provided to make predictions on new employee data.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/employee-performance-prediction.git
   cd employee-performance-prediction
   ```

2. **Install the necessary packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   - Start by loading and preprocessing the data in `data_processing.ipynb`.
   - Perform exploratory data analysis in `data_exploratory_analysis.ipynb`.
   - Train models in `train_model.ipynb`.
   - Use `predict_model.ipynb` to make predictions on new data.

4. **Explore the visualizations**:
   - Open `visualize.ipynb` to see the visual representations of the data and model performance.
  
## [Streamlit App](https://predictemployeeperformance.streamlit.app/)
- This work has an accompanying Streamlit App for predicting the rating of employee performance.
- Click [here](https://predictemployeeperformance.streamlit.app/) to interact with it.



## Contributing

Feel free to fork this project, submit issues, or make pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- INX Future Inc. Employee Performance Dataset
- Relevant research papers and articles used in the analysis are stored in the `references` directory.
```

This updated README includes the new background information and provides a comprehensive overview of the project, its objectives, structure, methodology, and instructions for running the project.
