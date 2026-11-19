import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR / "data.csv"

st.set_page_config(
    page_title="Student Academic Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

try:
    model_data = load_model()
    data = load_data()

    model = model_data
    feature_columns = list(model.feature_names_in_)

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

st.title("🎓 Student Academic Risk Predictor")

st.write(
    "Predict student academic outcomes using Machine Learning."
)

st.info(
    "Enter student information below to predict the academic outcome."
)

st.subheader("Student Assessment Form")

input_values = {}

raw_columns = data.drop(columns=["Target"]).columns

with st.form("student_form"):

    col1, col2 = st.columns(2)

    for index, column in enumerate(raw_columns):

        with col1 if index % 2 == 0 else col2:

            minimum = float(data[column].min())
            maximum = float(data[column].max())
            default = float(data[column].median())

            if pd.api.types.is_integer_dtype(data[column]):
                step = 1.0
            else:
                step = 0.01

            input_values[column] = st.number_input(
                column.strip(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step
            )

    submitted = st.form_submit_button(
        "🔍 Predict Student Outcome",
        use_container_width=True
    )

if submitted:

    input_df = pd.DataFrame([input_values])

    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == "Dropout":
            st.error("⚠️ Predicted Outcome: DROPOUT")

        elif prediction == "Graduate":
            st.success("🎓 Predicted Outcome: GRADUATE")

        elif prediction == "Enrolled":
            st.info("📚 Predicted Outcome: ENROLLED")

        else:
            st.success(f"Predicted Outcome: {prediction}")

    except Exception as e:
        st.error(f"Prediction error: {e}")