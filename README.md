# Employee Performance Rating

![image](https://github.com/user-attachments/assets/1e782a8a-7ee8-4fd6-bed9-832f8656bb8a)


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
│   ├── Requirement
|   ├── app.py
│   ├── Analysis
│   └── Summary
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
└── references                               # Reference documents and research papers
```

## Methodology

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
