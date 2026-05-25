# ============================================================
# TAMIL NADU ELECTION AI PREDICTION SYSTEM
# BEST STREAMLIT UI + ML + AUTO PREDICTION
# ============================================================

# RUN:
# streamlit run app.py

# INSTALL:
# pip install streamlit pandas numpy scikit-learn xgboost plotly joblib streamlit-lottie requests

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from streamlit_lottie import st_lottie

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tamil Nadu Election AI",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(135deg, #000428, #004e92);
    color: white;
}

/* TITLE */

.main-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #FFD700;
    text-shadow: 2px 2px 15px rgba(255,215,0,0.8);
}

/* SUBTITLE */

.sub-title {
    text-align: center;
    font-size: 22px;
    color: white;
    margin-bottom: 30px;
}

/* GLASS CARD */

.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(255,255,255,0.15);
}

/* METRIC BOX */

.metric-box {
    background: linear-gradient(135deg,#1f4037,#99f2c8);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: black;
    font-weight: bold;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.2);
}

/* BUTTON */

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#FFD700,#FFA500);
    color: black;
    border-radius: 15px;
    height: 60px;
    font-size: 22px;
    font-weight: bold;
    border: none;
    transition: 0.4s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 20px gold;
}

/* INPUT BOX */

.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141e30,#243b55);
}

/* HIDE STREAMLIT */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOTTIE ANIMATION
# ============================================================

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_puciaact.json"
)

# ============================================================
# TITLE
# ============================================================

st.markdown("<h1 class='main-title'>🗳️ Tamil Nadu Election AI</h1>", unsafe_allow_html=True)

st.markdown("<p class='sub-title'>Advanced Machine Learning Election Prediction Dashboard</p>", unsafe_allow_html=True)

# ============================================================
# ANIMATION
# ============================================================

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st_lottie(
        lottie_animation,
        height=300,
        key="election"
    )

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("eci_results_tamilnadu_2026.csv")

# ============================================================
# CLEAN DATA
# ============================================================

df.dropna(inplace=True)

# ============================================================
# CREATE WINNER COLUMN
# ============================================================

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# ============================================================
# ENCODING
# ============================================================

party_encoder = LabelEncoder()
const_encoder = LabelEncoder()

df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df['Vote_Strength'] = df['EVM Votes'] + df['Postal Votes']

df['Vote_Share'] = (
    df['Total Votes'] /
    df.groupby('Constituency')['Total Votes'].transform('sum')
)

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Vote_Strength',
    'Vote_Share',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# MODEL ACCURACY
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# ============================================================
# METRICS
# ============================================================

st.markdown("## 📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
    <h2>Accuracy</h2>
    <h1>{accuracy*100:.2f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    winners = df['Winner'].sum()

    st.markdown(f"""
    <div class="metric-box">
    <h2>Total Winners</h2>
    <h1>{winners}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_parties = df['Party'].nunique()

    st.markdown(f"""
    <div class="metric-box">
    <h2>Total Parties</h2>
    <h1>{total_parties}</h1>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PARTY PERFORMANCE CHART
# ============================================================

st.markdown("## 🏆 Top Political Parties")

party_votes = (
    df.groupby('Party')['Total Votes']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

party_df = pd.DataFrame({
    'Party': party_votes.index,
    'Votes': party_votes.values
})

fig = px.bar(
    party_df,
    x='Party',
    y='Votes',
    color='Votes',
    text='Votes',
    template='plotly_dark'
)

fig.update_layout(
    height=500,
    font_size=16
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("## 🔥 Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

fig2 = px.bar(
    importance,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    template='plotly_dark'
)

fig2.update_layout(height=500)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# AUTOMATIC WINNER PREDICTION
# ============================================================

st.markdown("## 🤖 Automatic Election Prediction")

latest = df.sort_values(
    by='Total Votes',
    ascending=False
).head(1)

winner_party = latest['Party'].values[0]
winner_candidate = latest['Candidate'].values[0]
winner_const = latest['Constituency'].values[0]
winner_votes = latest['Total Votes'].values[0]

prediction_input = pd.DataFrame({
    'EVM Votes': [latest['EVM Votes'].values[0]],
    'Postal Votes': [latest['Postal Votes'].values[0]],
    'Total Votes': [latest['Total Votes'].values[0]],
    '% Votes': [latest['% Votes'].values[0]],
    'Vote_Strength': [latest['Vote_Strength'].values[0]],
    'Vote_Share': [latest['Vote_Share'].values[0]],
    'Party_Encoded': [latest['Party_Encoded'].values[0]],
    'Constituency_Encoded': [latest['Constituency_Encoded'].values[0]]
})

auto_prediction = model.predict(prediction_input)[0]

probability = model.predict_proba(prediction_input)[0][1]

# ============================================================
# DISPLAY WINNER
# ============================================================

if auto_prediction == 1:

    st.success(f"""
    🏆 Predicted Winning Party: {winner_party}
    
    👤 Candidate: {winner_candidate}
    
    📍 Constituency: {winner_const}
    
    🗳️ Votes: {winner_votes}
    
    📈 Winning Probability: {probability*100:.2f}%
    """)

    st.balloons()

else:

    st.error("No Winner Predicted")

# ============================================================
# SIDEBAR MANUAL PREDICTION
# ============================================================

st.sidebar.title("🧠 Manual Prediction")

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
    value=50000
)

postal_votes = st.sidebar.number_input(
    "Postal Votes",
    value=500
)

total_votes = st.sidebar.number_input(
    "Total Votes",
    value=50500
)

vote_percent = st.sidebar.number_input(
    "% Votes",
    value=45.0
)

# ============================================================
# MANUAL PREDICTION
# ============================================================

if st.sidebar.button("Predict Result"):

    party_encoded = party_encoder.transform([party_name])[0]

    const_encoded = const_encoder.transform([const_name])[0]

    vote_strength = evm_votes + postal_votes

    sample = pd.DataFrame({
        'EVM Votes': [evm_votes],
        'Postal Votes': [postal_votes],
        'Total Votes': [total_votes],
        '% Votes': [vote_percent],
        'Vote_Strength': [vote_strength],
        'Vote_Share': [0.45],
        'Party_Encoded': [party_encoded],
        'Constituency_Encoded': [const_encoded]
    })

    pred = model.predict(sample)[0]

    prob = model.predict_proba(sample)[0][1]

    if pred == 1:

        st.sidebar.success(f"""
        🏆 WINNER
        
        Probability: {prob*100:.2f}%
        """)

    else:

        st.sidebar.error(f"""
        ❌ NOT WINNER
        
        Probability: {prob*100:.2f}%
        """)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "tn_election_model.pkl")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<hr>

<center>

<h3 style='color:gold;'>
🚀 Powered by XGBoost + Streamlit + AI
</h3>

</center>
""", unsafe_allow_html=True)
