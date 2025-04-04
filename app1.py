import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import time
from transformers import pipeline
import os

# Set page configuration
st.set_page_config(page_title="Emotion & Stress Detector", layout="centered")

# Load Pretrained Emotion Detection Model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Load ML model for emotion detection
pipe_lr = joblib.load(open("/workspaces/Emotion-zidio/svm_model1.pkl", "rb"))

# Stress dataset path
STRESS_CSV = "stress_log.csv"

# Emotion dictionary with emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Function to predict emotion
def predict_emotions(docx):
    result = pipe_lr.predict([docx])
    return result[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function to classify stress level
def classify_stress(text):
    if not text.strip():
        return None

    predictions = emotion_classifier(text)[0]
    
    # Define words that should indicate high stress
    high_stress_keywords = ["dying", "death", "panic", "anxiety", "suicide", "hopeless", "fearful", "danger"]

    # Check if the text contains high-stress words
    text_lower = text.lower()
    if any(word in text_lower for word in high_stress_keywords):
        return "Severe"

    # Calculate stress score based on emotion model
    stress_score = sum([p['score'] for p in predictions if p['label'] in ['fear', 'anger', 'sadness']])

    # Adjusted stress level thresholds
    if stress_score < 0.3:
        return "No Stress"
    elif stress_score < 0.6:
        return "Mild"
    elif stress_score < 0.85:
        return "Moderate"
    else:
        return "Severe"

# Function to recommend tasks based on emotion
def recommend_task(emotion):
    task_rules = {
        "fear": ["Practice deep breathing", "Talk to a mentor for reassurance"],
        "happy": ["Focus on creative tasks", "Collaborate with team"],
        "sad": ["Listen to a motivational podcast", "Review recent achievements"],
        "neutral": ["Prioritize high-priority tasks", "Schedule a team check-in"]
    }
    return task_rules.get(emotion, ["No recommendation available"])

# Function to log stress level in a dataset
def log_stress(user_id, text, stress_level):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    new_entry = pd.DataFrame([[user_id, timestamp, text, stress_level]], 
                             columns=["user_id", "timestamp", "text", "stress_level"])
    
    if os.path.exists(STRESS_CSV):
        df = pd.read_csv(STRESS_CSV)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(STRESS_CSV, index=False)

# Function to analyze stress history and trends
def analyze_stress_history(user_id):
    if not os.path.exists(STRESS_CSV):
        return None

    df = pd.read_csv(STRESS_CSV)
    user_data = df[df["user_id"] == user_id]

    if user_data.empty:
        return None

    stress_counts = user_data["stress_level"].value_counts().reset_index()
    stress_counts.columns = ["stress_level", "count"]

    return stress_counts

# Streamlit UI - Tabs for different features
st.title("🧠 Emotion & Stress Analysis")

tab1, tab2 , tab3= st.tabs(["😃 Emotion Detection", "🧘 Stress Level Detector", "📊 User Stress History"])

# Tab 1: Emotion Detection
with tab1:
    st.subheader("Analyze Your Emotion & Get Task Recommendations")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type Here", placeholder="Enter your thoughts...")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Predicted Emotion")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"{prediction} {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

            # Task Recommendation
            st.success("Recommended Actions")
            recommendations = recommend_task(prediction)
            for task in recommendations:
                st.write(f"✅ {task}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

# Tab 2: Stress Level Detector
with tab2:
    st.subheader("Analyze Your Stress Level")

    user_id = st.text_input("Enter User ID (for tracking stress history)", placeholder="e.g., user123")
    user_text = st.text_area("Enter your thoughts:", placeholder="Type here...")

    # Analyze Stress Button
    if st.button("Analyze Stress"):
        if user_id.strip() == "":
            st.error("⚠️ Please enter a User ID to track your stress history.")
        else:
            stress_level = classify_stress(user_text)

            if stress_level:
                if stress_level == "Beginning":
                    st.warning("⚠️ Mild Stress Detected: Take a short break and breathe.")
                elif stress_level == "Middle":
                    st.error("🚨 Moderate Stress Alert: Consider relaxation techniques like meditation.")
                elif stress_level == "Extreme":
                    st.error("🆘 Extreme Stress Alert: Seek support or talk to someone immediately!")

                # Log stress level
                log_stress(user_id, user_text, stress_level)

                # Show stress history
                stress_history = analyze_stress_history(user_id)
                if stress_history is not None:
                    st.subheader("📊 Your Stress Level History")
                    st.bar_chart(stress_history.set_index("stress_level"))
                else:
                    st.info("ℹ️ No stress history found for this user.")

            else:
                st.info("ℹ️ Enter some text to analyze your stress level.")
# Tab 3: Stress History Chart
with tab3:
    st.subheader("📊 User Stress History")

    # User ID Input
    history_user_id = st.text_input("Enter User ID to View Stress History", placeholder="e.g., user123")

    if st.button("Show Stress History"):
        if history_user_id.strip() == "":
            st.error("⚠️ Please enter a valid User ID.")
        else:
            # Load stress log
            if os.path.exists(STRESS_CSV):
                df = pd.read_csv(STRESS_CSV)

                # Filter records for entered User ID
                user_data = df[df["user_id"] == history_user_id]

                if user_data.empty:
                    st.info("ℹ️ No stress history found for this user.")
                else:
                    st.success(f"Stress history for **{history_user_id}**")
                    
                    # Convert timestamp to datetime format
                    user_data["timestamp"] = pd.to_datetime(user_data["timestamp"])

                    # Display Table
                    st.dataframe(user_data)

                    # Plot Stress Trends Over Time
                    chart = alt.Chart(user_data).mark_line().encode(
                        x='timestamp:T',
                        y=alt.Y('stress_level:N', sort=['Beginning', 'Middle', 'Extreme']),
                        color='stress_level',
                        tooltip=['timestamp', 'stress_level']
                    ).properties(title="Stress Levels Over Time")

                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("No stress records found. Start by analyzing stress in the previous tab.")
