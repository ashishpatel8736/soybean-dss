import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ✅ Set up Streamlit page (must be the first Streamlit command)
st.set_page_config(page_title="Soybean DSS", layout="wide")

# ✅ Load trained model and scaler
@st.cache_data
def load_model():
    try:
        model = joblib.load("model/soybean_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, scaler = load_model()
if model is None or scaler is None:
    st.stop()  # Stop if model loading fails

# ✅ Streamlit App Title & Description
st.title("🌱 Soybean Decision Support System (DSS)")
st.markdown(
    """
    Predict **soybean yield** and **protein content** based on key agricultural parameters.

    **How to interpret the results:**
    - A **high yield** indicates better crop productivity.
    - A **balanced protein content** is crucial for soybean nutritional quality.
    - Use these insights to adjust farming practices or inputs for optimal results.
    """
)

# 🎛 **Sidebar Input Form & Presets**
st.sidebar.header("🔢 Input Parameters")
preset_options = {
    "High Yield, High Protein": {
        "Plant Height (cm)": 90.0,
        "Number of Pods": 90,
        "Biological Weight (g)": 480.0,
        "Sugars (%)": 7.0,
        "Relative Water Content (%)": 85.0,
        "Chlorophyll A (mg/g)": 8.0,
        "Chlorophyll B (mg/g)": 7.5,
        "Protein Percentage (%)": 40.0,
        "Weight of 300 Seeds (g)": 480.0,
        "Leaf Area Index": 4.5,
        "Number of Seeds per Pod": 4.5
    },
    "Average Yield, Moderate Protein": {
        "Plant Height (cm)": 60.0,
        "Number of Pods": 50,
        "Biological Weight (g)": 300.0,
        "Sugars (%)": 5.0,
        "Relative Water Content (%)": 75.0,
        "Chlorophyll A (mg/g)": 5.0,
        "Chlorophyll B (mg/g)": 5.0,
        "Protein Percentage (%)": 25.0,
        "Weight of 300 Seeds (g)": 300.0,
        "Leaf Area Index": 2.5,
        "Number of Seeds per Pod": 3.0
    },
    "Low Yield, Low Protein": {
        "Plant Height (cm)": 40.0,
        "Number of Pods": 20,
        "Biological Weight (g)": 150.0,
        "Sugars (%)": 3.0,
        "Relative Water Content (%)": 60.0,
        "Chlorophyll A (mg/g)": 2.0,
        "Chlorophyll B (mg/g)": 2.0,
        "Protein Percentage (%)": 12.0,
        "Weight of 300 Seeds (g)": 150.0,
        "Leaf Area Index": 1.0,
        "Number of Seeds per Pod": 1.5
    },
    "Custom": None
}
preset_choice = st.sidebar.selectbox("Select a Preset", list(preset_options.keys()))

def user_input(preset_choice):
    if preset_choice != "Custom":
        preset = preset_options[preset_choice]
        st.sidebar.write("Using preset values:")
        for key, value in preset.items():
            st.sidebar.write(f"**{key}:** {value}")
        return np.array([[
            preset["Plant Height (cm)"],
            preset["Number of Pods"],
            preset["Biological Weight (g)"],
            preset["Sugars (%)"],
            preset["Relative Water Content (%)"],
            preset["Chlorophyll A (mg/g)"],
            preset["Chlorophyll B (mg/g)"],
            preset["Protein Percentage (%)"],
            preset["Weight of 300 Seeds (g)"],
            preset["Leaf Area Index"],
            preset["Number of Seeds per Pod"]
        ]])
    else:
        plant_height = st.sidebar.slider("🌿 Plant Height (cm)", 30.0, 100.0, 50.0)
        num_pods = st.sidebar.slider("🌾 Number of Pods", 5, 100, 50)
        bio_weight = st.sidebar.slider("🪴 Biological Weight (g)", 10.0, 500.0, 250.0)
        sugars = st.sidebar.slider("🍬 Sugars (%)", 0.1, 10.0, 5.0)
        rwc = st.sidebar.slider("💧 Relative Water Content (%)", 50.0, 100.0, 75.0)
        chlorophyll_a = st.sidebar.slider("🌿 Chlorophyll A (mg/g)", 0.1, 10.0, 5.0)
        chlorophyll_b = st.sidebar.slider("🌱 Chlorophyll B (mg/g)", 0.1, 10.0, 5.0)
        protein_pct = st.sidebar.slider("💪 Protein Percentage (%)", 10.0, 50.0, 25.0)
        seed_weight = st.sidebar.slider("⚖️ Weight of 300 Seeds (g)", 50.0, 500.0, 250.0)
        leaf_index = st.sidebar.slider("🍃 Leaf Area Index", 0.5, 5.0, 2.5)
        num_seeds_pod = st.sidebar.slider("🌰 Number of Seeds per Pod", 1.0, 5.0, 2.5)
        
        return np.array([[plant_height, num_pods, bio_weight, sugars, rwc,
                          chlorophyll_a, chlorophyll_b, protein_pct, seed_weight,
                          leaf_index, num_seeds_pod]])

data = user_input(preset_choice)

# ✅ Scale input data
try:
    scaled_data = scaler.transform(data)
except ValueError as e:
    st.error(f"❌ Error in feature scaling: {e}")
    st.stop()

# ✅ Make predictions
try:
    prediction = model.predict(scaled_data)
    st.write("🔍 **Raw Prediction Output:**", prediction)
    
    if prediction.ndim == 2 and prediction.shape[1] == 2:
        yield_pred, protein_pred = prediction[0]
    elif prediction.ndim == 1 and len(prediction) == 2:
        yield_pred, protein_pred = prediction
    elif prediction.ndim == 1:
        yield_pred = prediction[0]
        protein_pred = None
    else:
        st.error(f"⚠️ Unexpected prediction shape: {prediction.shape}. Check model output.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
    st.stop()

# ✅ Interpretation Functions with Tips
def interpret_yield(y):
    if y > 5000:
        return "Tip 💡: Excellent yield! Your crop productivity is high."
    elif y > 3000:
        return "Tip 💡: Good yield. Your crop productivity is above average."
    else:
        return "Tip 💡: Yield is low. Consider reviewing your cultivation practices."

def interpret_protein(p):
    if p is None:
        return "Tip 💡: Protein prediction not available."
    # Revised thresholds: lower values indicate higher protein content
    if p <= 0.7:
        return "Tip 💡: High protein content, excellent for nutritional quality."
    elif p <= 1.2:
        return "Tip 💡: Moderate protein content, generally acceptable."
    else:
        return "Tip 💡: Low protein content. Consider adjusting fertilization or variety."

yield_interpretation = interpret_yield(yield_pred)
protein_interpretation = interpret_protein(protein_pred)

# ✅ Display Prediction Results with Interpretations
st.subheader("📊 Prediction Results")
col1, col2 = st.columns(2)
col1.metric("🌾 Predicted Yield", f"{yield_pred:.2f} kg/ha", "Optimal")
col1.write(yield_interpretation)
if protein_pred is not None:
    col2.metric("💪 Predicted Protein Content", f"{protein_pred:.2f}%", "Balanced")
    col2.write(protein_interpretation)
else:
    col2.warning("⚠️ Protein prediction not available")

# ✅ Visualization
st.subheader("📈 Data Visualization")
chart_data = pd.DataFrame({
    "Parameter": ["Yield", "Protein"] if protein_pred is not None else ["Yield"],
    "Value": [yield_pred, protein_pred] if protein_pred is not None else [yield_pred]
})
fig = px.bar(chart_data, x="Parameter", y="Value", text_auto=True, color="Parameter",
             title="Soybean Yield & Protein Predictions",
             labels={"Value": "Prediction Value"})
st.plotly_chart(fig)

st.success("✅ Prediction completed! Adjust inputs or select a different preset to see changes.")
