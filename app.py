# =========================================================
# TAMIL NADU ELECTION PREDICTION
# ML PROJECT
# =========================================================

# ---------------- INSTALL REQUIRED LIBRARIES ----------------
# pip install streamlit pandas numpy matplotlib seaborn plotly
# pip install scikit-learn xgboost tensorflow joblib
# pip install vaderSentiment transformers torch

# ---------------- IMPORT LIBRARIES ----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* Header */
.main-title {
    font-size: 50px;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
    padding-top: 10px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 4px 25px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 0 25px rgba(124,58,237,0.5);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1f2937
    );
}

/* Input Labels */
label {
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Inputs */
.stNumberInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #475569 !important;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1e293b !important;
    color: white !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #2563eb, #9333ea);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #1d4ed8, #7e22ce);
}

/* Success */
.stSuccess {
    background-color: rgba(34,197,94,0.2) !important;
    color: white !important;
}

/* Error */
.stError {
    background-color: rgba(239,68,68,0.2) !important;
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
}

/* Animation */
.glow {
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #3b82f6;
    }
    to {
        text-shadow: 0 0 25px #8b5cf6;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title glow">
🗳️ Tamil Nadu Election Prediction
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("eci_results_tamilnadu_2026.csv")
    return df

df = load_data()

# =========================================================
# DATA CLEANING
# =========================================================

df = df.dropna()

df['Constituency'] = df['Constituency'].astype(str).str.strip()

# Winner Column
max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(
    df['Total Votes'] == max_votes,
    1,
    0
)

# =========================================================
# ENCODING
# =========================================================

party_encoder = LabelEncoder()
const_encoder = LabelEncoder()

df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# =========================================================
# FEATURES
# =========================================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODELS
# =========================================================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# =========================================================
# ACCURACY
# =========================================================

lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Prediction Panel")

party_name = st.sidebar.selectbox(
    "Select Party",
    sorted(df['Party'].unique())
)

constituency_name = st.sidebar.selectbox(
    "Select Constituency",
    sorted(df['Constituency'].unique())
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

vote_percent = st.sidebar.slider(
    "Vote Percentage",
    0.0,
    100.0,
    45.0
)

# =========================================================
# ENCODE USER INPUT
# =========================================================

party_encoded = party_encoder.transform([party_name])[0]
const_encoded = const_encoder.transform([constituency_name])[0]

sample_data = pd.DataFrame({
    'EVM Votes': [evm_votes],
    'Postal Votes': [postal_votes],
    'Total Votes': [total_votes],
    '% Votes': [vote_percent],
    'Party_Encoded': [party_encoded],
    'Constituency_Encoded': [const_encoded]
})

# =========================================================
# PREDICTION
# =========================================================

prediction = xgb_model.predict(sample_data)[0]
probability = xgb_model.predict_proba(sample_data)[0][1]

# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
    <h2>{round(lr_acc*100,2)}%</h2>
    <p>Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
    <h2>{round(rf_acc*100,2)}%</h2>
    <p>Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
    <h2>{round(xgb_acc*100,2)}%</h2>
    <p>XGBoost Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# RESULT SECTION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if prediction == 1:
    st.success(f"""
    🏆 HIGH WINNING CHANCE
    
    Winning Probability: {round(probability*100,2)}%
    """)
else:
    st.error(f"""
    ❌ LOW WINNING CHANCE
    
    Winning Probability: {round(probability*100,2)}%
    """)

# =========================================================
# MODEL COMPARISON CHART
# =========================================================

st.markdown("## 📊 Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_acc, rf_acc, xgb_acc]
})

fig = px.bar(
    accuracy_df,
    x='Model',
    y='Accuracy',
    text='Accuracy',
    template='plotly_dark',
    height=500
)

fig.update_traces(texttemplate='%{text:.2f}')
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PARTY VOTES CHART
# =========================================================

st.markdown("## 🗳️ Top Parties by Votes")

party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

fig2 = px.bar(
    x=party_votes.index,
    y=party_votes.values,
    template='plotly_dark',
    labels={'x':'Party', 'y':'Votes'},
    height=500
)

fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.markdown("## 🔥 Confusion Matrix")

cm = confusion_matrix(y_test, xgb_model.predict(X_test))

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
# FEATURE IMPORTANCE
# =========================================================

st.markdown("## ⭐ Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

fig4 = px.bar(
    importance,
    x='Importance',
    y='Feature',
    orientation='h',
    template='plotly_dark',
    height=500
)

fig4.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig4, use_container_width=True)

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

st.markdown("## 😊 Public Sentiment Analysis")

tweets = [
    "MK Stalin is doing good work",
    "Bad government performance",
    "People support DMK",
    "Corruption issue increasing",
    "Excellent development in Tamil Nadu"
]

df_tweets = pd.DataFrame(tweets, columns=['Tweet'])

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = analyzer.polarity_scores(text)

    if score['compound'] >= 0.05:
        return 'Positive'

    elif score['compound'] <= -0.05:
        return 'Negative'

    else:
        return 'Neutral'

df_tweets['Sentiment'] = df_tweets['Tweet'].apply(get_sentiment)

st.dataframe(df_tweets)

sentiment_count = df_tweets['Sentiment'].value_counts()

fig5 = px.pie(
    names=sentiment_count.index,
    values=sentiment_count.values,
    template='plotly_dark',
    height=450
)

fig5.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)

st.plotly_chart(fig5, use_container_width=True)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.markdown("## 📄 Classification Report")

report = classification_report(
    y_test,
    xgb_model.predict(X_test),
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(xgb_model, 'tamilnadu_election_model.pkl')

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br><br>
<center>
<h4 style='color:white;'>
🚀 Developed with Streamlit • XGBoost • AI • Machine Learning
</h4>
</center>
""", unsafe_allow_html=True)
