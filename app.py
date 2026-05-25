# ============================================================
# 🗳️ TAMIL NADU ELECTION PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

/* GOOGLE FONT */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ============================================================
MAIN BACKGROUND
============================================================ */

.stApp {

    background:
    linear-gradient(
        rgba(0,0,0,0.82),
        rgba(0,0,0,0.88)
    ),

    url("https://images.unsplash.com/photo-1541872705-1f73c6400ec9?q=80&w=2070");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* HIDE STREAMLIT */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ============================================================
TITLE
============================================================ */

.main-title {

    text-align:center;

    font-size:70px;

    font-weight:900;

    background: linear-gradient(
        90deg,
        #FFD700,
        #FFA500,
        #FF4500
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    text-shadow:
        0px 0px 18px rgba(255,215,0,0.45);

    margin-top:10px;
}

.sub-title {

    text-align:center;

    color:#F5F5F5;

    font-size:22px;

    margin-bottom:35px;
}

/* ============================================================
SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111111,
        #1a1a1a
    );

    border-right:1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TITLE */

.sidebar-title {

    color:#FFD700;

    text-align:center;

    font-size:30px;

    font-weight:bold;

    margin-bottom:20px;
}

/* ============================================================
METRICS
============================================================ */

[data-testid="metric-container"] {

    background:
    rgba(25,25,25,0.92);

    border-radius:22px;

    padding:18px;

    border:1px solid rgba(255,255,255,0.08);

    box-shadow:
        0px 10px 25px rgba(0,0,0,0.45);

    transition:0.3s;
}

[data-testid="metric-container"]:hover {

    transform:translateY(-5px);

    border:1px solid gold;

    box-shadow:
        0px 0px 20px rgba(255,215,0,0.35);
}

/* METRIC TEXT */

[data-testid="metric-container"] label {

    color:#FFD700 !important;

    font-size:18px !important;
}

[data-testid="metric-container"] div {

    color:white !important;

    font-weight:bold;
}

/* ============================================================
INPUTS
============================================================ */

.stNumberInput input {

    background-color:#1e1e1e !important;

    color:white !important;

    border:2px solid #444 !important;

    border-radius:12px !important;

    font-size:18px !important;
}

/* SELECT BOX */

.stSelectbox div[data-baseweb="select"] {

    background-color:#1e1e1e !important;

    border-radius:12px !important;

    border:2px solid #444 !important;
}

.stSelectbox * {
    color:white !important;
}

/* SLIDER */

.stSlider * {
    color:white !important;
}

/* ============================================================
BUTTON
============================================================ */

.stButton>button {

    width:100%;

    background:
    linear-gradient(
        90deg,
        #ff416c,
        #ff4b2b
    );

    color:white;

    border:none;

    border-radius:15px;

    padding:14px;

    font-size:20px;

    font-weight:bold;

    box-shadow:
        0px 8px 20px rgba(255,75,43,0.45);

    transition:0.3s;
}

.stButton>button:hover {

    transform:scale(1.03);

    background:
    linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );
}

/* ============================================================
TEXT
============================================================ */

h1,h2,h3,h4,h5,h6 {

    color:#FFD700 !important;
}

p, label, span, div {

    color:white;
}

/* ============================================================
DATAFRAME
============================================================ */

[data-testid="stDataFrame"] {

    background:rgba(20,20,20,0.88);

    border-radius:18px;

    padding:10px;
}

/* ============================================================
WINNER BOX
============================================================ */

.winner-box {

    background:
    linear-gradient(
        135deg,
        rgba(0,0,0,0.96),
        rgba(15,15,15,0.96)
    ) !important;

    border:4px solid #FFD700;

    border-radius:30px;

    padding:50px;

    text-align:center;

    margin-top:25px;

    box-shadow:
        0px 0px 40px rgba(255,215,0,0.7);

    animation:pulse 2s infinite;
}

/* MAIN TITLE */

.winner-box h1 {

    color:#FFD700 !important;

    font-size:52px !important;

    font-weight:900 !important;

    text-shadow:
        0px 0px 18px rgba(255,215,0,0.8);

    margin-bottom:25px;
}

/* CANDIDATE NAME */

.winner-box h2 {

    color:#FFFFFF !important;

    font-size:40px !important;

    font-weight:800 !important;

    text-shadow:
        0px 0px 15px rgba(255,255,255,0.5);

    margin-top:15px;
}

/* PARTY NAME */

.winner-box h3 {

    color:#00FFCC !important;

    font-size:32px !important;

    font-weight:700 !important;

    text-shadow:
        0px 0px 15px rgba(0,255,204,0.7);

    margin-top:12px;
}

/* PULSE EFFECT */

@keyframes pulse {

    0% {
        transform:scale(1);
    }

    50% {
        transform:scale(1.02);
    }

    100% {
        transform:scale(1);
    }
}

/* ============================================================
PROGRESS BAR
============================================================ */

.stProgress > div > div > div > div {

    background:
    linear-gradient(
        90deg,
        #FFD700,
        #ff6600
    );
}

/* ============================================================
FOOTER
============================================================ */

.footer {

    text-align:center;

    margin-top:40px;

    color:#cccccc;

    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class="main-title">
🗳️ Tamil Nadu Election AI Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
AI Powered Election Prediction • Political Analytics • 3D Celebration UI
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("eci_results_tamilnadu_2026.csv")

df = load_data()

df = df.dropna()

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
# LABEL ENCODING
# ============================================================

party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# ============================================================
# FEATURES
# ============================================================

X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# ============================================================
# TRAIN MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

joblib.dump(model, "tamilnadu_model.pkl")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div class="sidebar-title">
🗳️ Navigation
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📄 Dataset",
        "📊 Analytics",
        "🏆 Constituency Winner",
        "🔮 Election Prediction",
        "📈 Sentiment Analysis"
    ]
)

# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Constituencies",
            df['Constituency'].nunique()
        )

    with col2:
        st.metric(
            "Total Candidates",
            df['Candidate'].nunique()
        )

    with col3:
        st.metric(
            "Total Parties",
            df['Party'].nunique()
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="winner-box">

        <h1>🇮🇳 Tamil Nadu Election Intelligence</h1>

        <h2>AI Based Prediction System</h2>

        <h3>Real-Time Political Analytics Dashboard</h3>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DATASET
# ============================================================

elif menu == "📄 Dataset":

    st.subheader("📄 Tamil Nadu Election Dataset")

    st.dataframe(df, use_container_width=True)

# ============================================================
# ANALYTICS
# ============================================================

elif menu == "📊 Analytics":

    st.subheader("📊 Election Analytics")

    # PARTY VOTES

    party_votes = df.groupby('Party')['Total Votes'] \
                    .sum() \
                    .sort_values(ascending=False) \
                    .head(10)

    fig = px.bar(
        x=party_votes.index,
        y=party_votes.values,
        color=party_votes.values,
        text=party_votes.values,
        title="Top 10 Parties by Votes"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # SEAT SHARE

    party_seats = df[df['Winner'] == 1]['Party'].value_counts()

    fig2 = px.pie(
        names=party_seats.index,
        values=party_seats.values,
        hole=0.45,
        title="Seat Share"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# CONSTITUENCY WINNER
# ============================================================

elif menu == "🏆 Constituency Winner":

    st.subheader("🏆 Find Constituency Winner")

    constituency = st.selectbox(
        "Select Constituency",
        sorted(df['Constituency'].unique())
    )

    const_data = df[df['Constituency'] == constituency]

    winner = const_data.loc[
        const_data['Total Votes'].idxmax()
    ]

    st.balloons()

    st.markdown(f"""
    <div class="winner-box">

        <h1>🎉 WINNER 🎉</h1>

        <h2>{winner['Candidate']}</h2>

        <h3>{winner['Party']}</h3>

        <h2>🏆 {int(winner['Total Votes'])} Votes</h2>

    </div>
    """, unsafe_allow_html=True)

    ranked = const_data.sort_values(
        by='Total Votes',
        ascending=False
    )

    st.subheader("📋 Candidate Ranking")

    st.dataframe(
        ranked[
            ['Candidate', 'Party', 'Total Votes', '% Votes']
        ],
        use_container_width=True
    )

# ============================================================
# ELECTION PREDICTION
# ============================================================

elif menu == "🔮 Election Prediction":

    st.subheader("🔮 Predict Future Election Winner")

    col1, col2 = st.columns(2)

    with col1:

        evm_votes = st.number_input(
            "EVM Votes",
            min_value=0
        )

        postal_votes = st.number_input(
            "Postal Votes",
            min_value=0
        )

        total_votes = st.number_input(
            "Total Votes",
            min_value=0
        )

    with col2:

        vote_percent = st.slider(
            "Vote Percentage",
            0.0,
            100.0,
            45.0
        )

        party = st.selectbox(
            "Select Party",
            sorted(df['Party'].unique())
        )

        constituency = st.selectbox(
            "Select Constituency",
            sorted(df['Constituency'].unique())
        )

    party_encoded = party_encoder.transform([party])[0]

    const_encoded = const_encoder.transform([constituency])[0]

    if st.button("🚀 Predict Winner"):

        sample = pd.DataFrame({

            'EVM Votes': [evm_votes],
            'Postal Votes': [postal_votes],
            'Total Votes': [total_votes],
            '% Votes': [vote_percent],
            'Party_Encoded': [party_encoded],
            'Constituency_Encoded': [const_encoded]

        })

        prediction = model.predict(sample)

        probability = model.predict_proba(sample)[0][1]

        if prediction[0] == 1:

            st.balloons()

            st.snow()

            st.markdown(f"""
            <div class="winner-box">

                <h1>🎊 WINNER PREDICTED 🎊</h1>

                <h2>{party}</h2>

                <h3>🔥 Massive Winning Probability</h3>

                <h1>{round(probability*100,2)}%</h1>

            </div>
            """, unsafe_allow_html=True)

            st.progress(float(probability))

        else:

            st.error("❌ Predicted Result: NOT WINNER")

            st.progress(float(probability))

            st.write(
                f"Winning Probability: {round(probability*100,2)}%"
            )

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif menu == "📈 Sentiment Analysis":

    st.subheader("📈 Political Sentiment Analysis")

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [65, 20, 15]
    })

    fig = px.bar(
        sentiment_data,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        text='Count',
        title='Public Sentiment Analysis'
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sentiment_data)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
🇮🇳 Tamil Nadu Election AI Dashboard • Built with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
