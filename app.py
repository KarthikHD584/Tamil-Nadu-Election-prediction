# ============================================
# TAMIL NADU ELECTION PREDICTION 
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #FFD700;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: white;
}

.metric-box {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.2);
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

.stNumberInput input {
    background-color: white !important;
    color: black !important;
}

.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.markdown("<h1 class='main-title'>🗳️ Tamil Nadu Election Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Machine Learning Based Election Winner Prediction System</p>", unsafe_allow_html=True)

# --------------------------------
# LOAD DATA
# --------------------------------
df = pd.read_csv("eci_results_tamilnadu_2026.csv")

# --------------------------------
# CLEAN DATA
# --------------------------------
df = df.dropna()

# --------------------------------
# CREATE WINNER COLUMN
# --------------------------------
max_votes = df.groupby('Constituency')['Total Votes'].transform('max')

df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

# --------------------------------
# ENCODING
# --------------------------------
party_encoder = LabelEncoder()
df['Party_Encoded'] = party_encoder.fit_transform(df['Party'])

const_encoder = LabelEncoder()
df['Constituency_Encoded'] = const_encoder.fit_transform(df['Constituency'])

# --------------------------------
# FEATURES & TARGET
# --------------------------------
X = df[[
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes',
    'Party_Encoded',
    'Constituency_Encoded'
]]

y = df['Winner']

# --------------------------------
# SPLIT DATA
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# TRAIN MODELS
# --------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# --------------------------------
# PREDICTIONS
# --------------------------------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# --------------------------------
# ACCURACY
# --------------------------------
lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

# --------------------------------
# METRICS
# --------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-box'>
        <h2>Logistic Regression</h2>
        <h1>{lr_acc:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-box'>
        <h2>Random Forest</h2>
        <h1>{rf_acc:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-box'>
        <h2>XGBoost</h2>
        <h1>{xgb_acc:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------------------------
# MODEL ACCURACY GRAPH
# --------------------------------
st.subheader("📊 Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_acc, rf_acc, xgb_acc]
})

fig = px.bar(
    accuracy_df,
    x='Model',
    y='Accuracy',
    text='Accuracy',
    color='Model',
    template='plotly_dark'
)

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# TOP PARTIES
# --------------------------------
st.subheader("🏆 Top 10 Parties by Total Votes")

party_votes = df.groupby('Party')['Total Votes'].sum().sort_values(ascending=False).head(10)

party_df = pd.DataFrame({
    'Party': party_votes.index,
    'Votes': party_votes.values
})

fig2 = px.bar(
    party_df,
    x='Party',
    y='Votes',
    color='Votes',
    template='plotly_dark'
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------
# CONFUSION MATRIX
# --------------------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, xgb_pred)

fig3, ax = plt.subplots(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig3)

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------
st.subheader("🔥 Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

fig4 = px.bar(
    importance,
    x='Feature',
    y='Importance',
    color='Importance',
    template='plotly_dark'
)

st.plotly_chart(fig4, use_container_width=True)

# --------------------------------
# PREDICTION SECTION
# --------------------------------
st.markdown("---")
st.subheader("🧠 Predict Future Election Winner")

col1, col2 = st.columns(2)

with col1:
    evm_votes = st.number_input("EVM Votes", value=70000)

    postal_votes = st.number_input("Postal Votes", value=500)

    total_votes = st.number_input("Total Votes", value=70500)

with col2:
    percent_votes = st.number_input("% Votes", value=40.2)

    party_name = st.selectbox(
        "Select Party",
        party_encoder.classes_
    )

    constituency_name = st.selectbox(
        "Select Constituency",
        const_encoder.classes_
    )

# --------------------------------
# ENCODE INPUTS
# --------------------------------
party_encoded = party_encoder.transform([party_name])[0]

const_encoded = const_encoder.transform([constituency_name])[0]

# --------------------------------
# PREDICT BUTTON
# --------------------------------
if st.button("🔍 Predict Winner"):

    sample_data = pd.DataFrame({
        'EVM Votes': [evm_votes],
        'Postal Votes': [postal_votes],
        'Total Votes': [total_votes],
        '% Votes': [percent_votes],
        'Party_Encoded': [party_encoded],
        'Constituency_Encoded': [const_encoded]
    })

    prediction = xgb_model.predict(sample_data)

    probability = xgb_model.predict_proba(sample_data)[0][1]

    st.markdown("---")

    if prediction[0] == 1:
        st.success(f"🏆 {party_name} is Predicted WINNER")
        st.balloons()
    else:
        st.error(f"❌ {party_name} is Predicted NOT WINNER")

    st.info(f"Winning Probability: {probability*100:.2f}%")

# --------------------------------
# CLASSIFICATION REPORT
# --------------------------------
st.subheader("📄 Classification Report")

report = classification_report(y_test, xgb_pred)

st.text(report)

# --------------------------------
# SAVE MODEL
# --------------------------------
joblib.dump(xgb_model, 'tamilnadu_election_model.pkl')

st.success("✅ Model Saved Successfully")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("""
<hr>
<center>
<h4>Developed using Streamlit + Machine Learning + XGBoost</h4>
</center>
""", unsafe_allow_html=True)
