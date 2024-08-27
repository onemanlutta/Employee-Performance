Requirements

### Key Libraries Required and Their Importance

1. **Pandas**
   - **Purpose**: Data manipulation and analysis.
   - **Importance**: Used for loading, cleaning, and preprocessing the dataset, making it essential for handling data frames and performing operations like merging, filtering, and grouping data.

2. **NumPy**
   - **Purpose**: Numerical computing.
   - **Importance**: Provides support for arrays and matrices, along with a large collection of mathematical functions to operate on these structures, which is crucial for data manipulation and mathematical computations.

3. **Scikit-learn**
   - **Purpose**: Machine learning.
   - **Importance**: Core library for implementing machine learning models. It includes tools for model selection, cross-validation, hyperparameter tuning (`GridSearchCV`), and metrics to evaluate model performance.

4. **Imbalanced-learn (imblearn)**
   - **Purpose**: Handling imbalanced datasets.
   - **Importance**: Contains the `SMOTE` algorithm, which is used to address class imbalance in the dataset, ensuring the model is trained on a balanced representation of the classes.

5. **SHAP (SHapley Additive exPlanations)**
   - **Purpose**: Model interpretability.
   - **Importance**: Provides methods to explain individual predictions and global model behavior, making it possible to understand and visualize how different features contribute to model decisions.

6. **Matplotlib & Seaborn**
   - **Purpose**: Data visualization.
   - **Importance**: Used for creating various plots (e.g., feature importance, SHAP summary, dependence, and force plots) that help in analyzing and interpreting both the data and the model’s behavior.

7. **Joblib**
   - **Purpose**: Model persistence.
   - **Importance**: Allows for saving and loading machine learning models efficiently, ensuring the model can be reused without retraining.

These libraries form the backbone of the project, enabling everything from data preprocessing to model training, evaluation, interpretability, and deployment.