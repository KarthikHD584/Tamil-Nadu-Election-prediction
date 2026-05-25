# =========================================================
# ADVANCED ELECTION AI DASHBOARD - STREAMLIT
# BEST UI + AUTO WINNER PREDICTION + MODERN DESIGN
# =========================================================

# RUN:
# streamlit run app.py

# INSTALL:
# pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn joblib

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Election AI Dashboard",
    page_icon="🗳️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(135deg,#141E30,#243B55);
    color: white;
}

/* TITLE */

.title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    color: #FFD700;
    text-shadow: 0px 0px 20px rgba(255,215,0,0.8);
}

/* SUBTITLE */

.subtitle {
    text-align: center;
    font-size: 22px;
    color: white;
    margin-bottom: 30px;
}

/* METRIC CARDS */

.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0px 0px 20px rgba(255,255,255,0.15);
    transition: 0.4s;
}

.metric-card:hover {
    transform: scale(1.03);
}

/* BUTTON */

.stButton>button {
    background: linear-gradient(90deg,#FFD700,#FFA500);
    color: black;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

/* INPUT BOX */

.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F2027,#203A43,#2C5364);
}

/* ANIMATION */

.glow {
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #FFD700;
    }
    to {
        text-shadow: 0 0 25px #FFD700;
    }
}

/* HIDE FOOTER */

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<h1 class='title glow'>🗳️ Election AI Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning Based Tamil Nadu Election Prediction System</p>",
    unsafe_allow_html=True
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("eci_results_tamilnadu_2026.csv")

# =========================================================
# DATA CLEANING
# =========================================================

df.dropna(inplace=True)

# =========================================================
# CREATE WINNER COLUMN
# =========================================================

max_votes = df.groupby("Constituency")["Total Votes"].transform("max")

df["Winner"] = np.where(
    df["Total Votes"] == max_votes,
    1,
    0
)

# =========================================================
# ENCODING
# =========================================================

party_encoder = LabelEncoder()
const_encoder = LabelEncoder()

df["Party_Encoded"] = party_encoder.fit_transform(df["Party"])

df["Constituency_Encoded"] = const_encoder.fit_transform(df["Constituency"])

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["Vote_Strength"] = (
    df["EVM Votes"] +
    df["Postal Votes"]
)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df[[
    "EVM Votes",
    "Postal Votes",
    "Total Votes",
    "% Votes",
    "Vote_Strength",
    "Party_Encoded",
    "Constituency_Encoded"
]]

y = df["Winner"]

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# MODEL PREDICTION
# =========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
    <h2>🎯 Accuracy</h2>
    <h1>{accuracy*100:.2f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_parties = df["Party"].nunique()

    st.markdown(f"""
    <div class="metric-card">
    <h2>🏛️ Parties</h2>
    <h1>{total_parties}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    winners = df["Winner"].sum()

    st.markdown(f"""
    <div class="metric-card">
    <h2>🏆 Winners</h2>
    <h1>{winners}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# AUTO WINNER PREDICTION
# =========================================================

st.subheader("🤖 Automatic Winner Prediction")

top_candidate = df.sort_values(
    by="Total Votes",
    ascending=False
).iloc[0]

winner_party = top_candidate["Party"]
winner_candidate = top_candidate["Candidate"]
winner_const = top_candidate["Constituency"]
winner_votes = top_candidate["Total Votes"]

st.success(f"""
🏆 Predicted Winning Party: {winner_party}

👤 Candidate: {winner_candidate}

📍 Constituency: {winner_const}

🗳️ Total Votes: {winner_votes}
""")

st.balloons()

# =========================================================
# PARTY ANALYSIS CHART
# =========================================================

st.subheader("📊 Top 10 Parties Vote Analysis")

party_votes = (
    df.groupby("Party")["Total Votes"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

party_df = pd.DataFrame({
    "Party": party_votes.index,
    "Votes": party_votes.values
})

fig = px.bar(
    party_df,
    x="Party",
    y="Votes",
    color="Votes",
    text="Votes",
    template="plotly_dark"
)

fig.update_layout(
    height=500,
    font_size=16
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🔥 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig2 = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    template="plotly_dark"
)

fig2.update_layout(height=500)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, predictions)

fig3, ax = plt.subplots(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig3)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    predictions
)

st.text(report)

# =========================================================
# MANUAL PREDICTION SIDEBAR
# =========================================================

st.sidebar.title("🧠 Predict Election Result")

party_name = st.sidebar.selectbox(
    "Select Party",
    party_encoder.classes_
)

const_name = st.sidebar.selectbox(
    "Select Constituency",
    const_encoder.classes_
)

evm_votes = st.sidebar.number_input(
    "EVM Votes",
    value=70000
)

postal_votes = st.sidebar.number_input(
    "Postal Votes",
    value=500
)

total_votes = st.sidebar.number_input(
    "Total Votes",
    value=70500
)

vote_percent = st.sidebar.number_input(
    "% Votes",
    value=45.0
)

# =========================================================
# MANUAL PREDICTION BUTTON
# =========================================================

if st.sidebar.button("Predict Winner"):

    party_encoded = party_encoder.transform([party_name])[0]

    const_encoded = const_encoder.transform([const_name])[0]

    vote_strength = evm_votes + postal_votes

    sample_data = pd.DataFrame({
        "EVM Votes": [evm_votes],
        "Postal Votes": [postal_votes],
        "Total Votes": [total_votes],
        "% Votes": [vote_percent],
        "Vote_Strength": [vote_strength],
        "Party_Encoded": [party_encoded],
        "Constituency_Encoded": [const_encoded]
    })

    prediction = model.predict(sample_data)[0]

    probability = model.predict_proba(sample_data)[0][1]

    if prediction == 1:

        st.sidebar.success(f"""
        🏆 WINNER
        
        Probability: {probability*100:.2f}%
        """)

    else:

        st.sidebar.error(f"""
        ❌ NOT WINNER
        
        Probability: {probability*100:.2f}%
        """)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "election_ai_model.pkl")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>

<h3 style='color:gold;'>
🚀 AI Powered Election Prediction System
</h3>

<p>
Built with Streamlit + XGBoost + Machine Learning
</p>

</center>
""", unsafe_allow_html=True)
