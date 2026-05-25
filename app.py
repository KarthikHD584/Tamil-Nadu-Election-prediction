# ==========================================
# PERFECT VISIBILITY PREMIUM CSS
# ==========================================

st.markdown("""
<style>

/* GOOGLE FONT */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* =========================================
MAIN BACKGROUND
========================================= */

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

/* =========================================
TITLE
========================================= */

.main-title {

    text-align:center;

    font-size:68px;

    font-weight:900;

    background: linear-gradient(
        90deg,
        #FFD700,
        #FFA500,
        #FF4500
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    margin-bottom:5px;

    text-shadow:
        0px 0px 18px rgba(255,215,0,0.45);
}

.sub-title {

    text-align:center;

    color:#F5F5F5;

    font-size:22px;

    margin-bottom:35px;

    font-weight:400;
}

/* =========================================
SIDEBAR
========================================= */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111111,
        #1b1b1b
    );

    border-right:1px solid rgba(255,255,255,0.08);
}

/* =========================================
METRICS
========================================= */

[data-testid="metric-container"] {

    background:
    rgba(25,25,25,0.85);

    border-radius:20px;

    padding:18px;

    border:1px solid rgba(255,255,255,0.08);

    box-shadow:
        0px 8px 20px rgba(0,0,0,0.4);
}

[data-testid="metric-container"] label {

    color:#FFD700 !important;

    font-size:18px !important;
}

[data-testid="metric-container"] div {

    color:white !important;

    font-weight:bold;
}

/* =========================================
INPUT BOXES
========================================= */

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

    color:white !important;

    border:2px solid #444 !important;
}

/* DROPDOWN TEXT */

.stSelectbox * {
    color:white !important;
}

/* SLIDER */

.stSlider * {
    color:white !important;
}

/* =========================================
BUTTON
========================================= */

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

    border-radius:14px;

    padding:14px;

    font-size:20px;

    font-weight:bold;

    box-shadow:
        0px 8px 22px rgba(255,75,43,0.45);

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

/* =========================================
HEADINGS
========================================= */

h1,h2,h3,h4,h5,h6 {

    color:#FFD700 !important;
}

/* NORMAL TEXT */

p, label, span {

    color:white !important;
}

/* =========================================
DATAFRAME
========================================= */

[data-testid="stDataFrame"] {

    background:rgba(20,20,20,0.88);

    border-radius:18px;

    padding:10px;
}

/* =========================================
WINNER BOX - FIXED VISIBILITY
========================================= */

.winner-box {

    background:
    linear-gradient(
        135deg,
        rgba(20,20,20,0.95),
        rgba(35,35,35,0.95)
    );

    border:3px solid #FFD700;

    border-radius:28px;

    padding:45px;

    text-align:center;

    margin-top:25px;

    box-shadow:
        0px 0px 35px rgba(255,215,0,0.45);

    animation:pulse 2s infinite;
}

/* WINNER TEXT */

.winner-box h1 {

    color:#FFD700 !important;

    font-size:48px !important;

    font-weight:900 !important;

    margin-bottom:20px;
}

.winner-box h2 {

    color:white !important;

    font-size:34px !important;

    font-weight:700 !important;

    margin-top:12px;
}

.winner-box h3 {

    color:#00ffcc !important;

    font-size:28px !important;

    margin-top:10px;
}

/* ANIMATION */

@keyframes pulse {

    0% {
        transform:scale(1);
    }

    50% {
        transform:scale(1.015);
    }

    100% {
        transform:scale(1);
    }
}

/* =========================================
PROGRESS BAR
========================================= */

.stProgress > div > div > div > div {

    background:
    linear-gradient(
        90deg,
        #FFD700,
        #ff6600
    );
}

/* =========================================
FOOTER
========================================= */

.footer {

    text-align:center;

    margin-top:40px;

    color:#cccccc;

    font-size:15px;
}

</style>
""", unsafe_allow_html=True)
