# 🎓 Student Dropout and Academic Success Prediction

## 1. Project Overview

This project aims to predict student academic outcomes using Machine Learning.

It identifies whether a student is likely to:

- Graduate
- Dropout
- Remain Enrolled

The main objective is to help identify students who may be at academic risk.

## 2. Dataset Information

- Total Records: 4,424
- Total Columns: 37
- Input Features: 36
- Target Column: Target
- Missing Values: 0
- Duplicate Records: 0

### Target Distribution

| Student Status | Count | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

## 3. Project Modules and Team Contributions

### Student 1 – Data Analysis and EDA
- Dataset loading and exploration
- Data quality checking
- Statistical analysis
- Target distribution visualization
- Feature histograms
- Correlation heatmap
- EDA report preparation

### Student 2 – Machine Learning Model Comparison
- Data preprocessing
- Feature preparation
- Model training
- Model comparison
- Performance evaluation

### Student 3 – Model Optimization
- Hyperparameter tuning
- Model optimization
- Cross-validation
- Classification report
- Optimized Random Forest model

### Student 4 – Streamlit Application
- Streamlit application development
- Student assessment form
- Machine Learning model integration
- Student outcome prediction
- Prediction result display

## 4. Machine Learning Model

The optimized model used in the application is:

**Random Forest Classifier**

Model Parameters:
- Number of Estimators: 200
- Maximum Depth: 20
- Random State: 42

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook
- Git and GitHub

## 6. Project Files

- `data.csv` – Dataset
- `01_data_analysis.ipynb` – Data analysis notebook
- `02_model_comparison.ipynb` – Model comparison notebook
- `app.py` – Streamlit application
- `best_model.pkl` – Optimized Machine Learning model
- `student_dropout_model.pkl` – Model comparison output
- `src/` – Source code files

## 7. How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/ismatkhan-dev/student-dropout-prediction.git
```

### Step 2: Open Project Folder

```bash
cd student-dropout-prediction
```

### Step 3: Install Required Libraries

```bash
python -m pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
```

### Step 4: Run Streamlit Application

```bash
python -m streamlit run app.py
```

### Step 5: Open Application

Open the local URL displayed in the terminal:

```text
http://localhost:8501
```

## 8. Project Status

The project includes:

- Data Analysis and EDA
- Machine Learning Model Comparison
- Model Optimization
- Streamlit Web Application
- Optimized Model Integration
- Student Outcome Prediction

## 9. Future Scope

- Improve prediction performance
- Add more data visualizations
- Deploy the application online
- Provide early academic risk alerts

## 10. Conclusion

This project demonstrates how Machine Learning can be used to predict student academic outcomes.

It provides a simple web application where users can enter student information and receive a predicted academic outcome.