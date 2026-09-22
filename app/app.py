import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from streamlit_option_menu import option_menu


# =========================================================
# HTML RENDER FUNCTION
# =========================================================

def render_html(content, unsafe_allow_html=True):

    # Streamlit versions which support st.html()
    if hasattr(st, "html"):
        st.html(content)

    # Older Streamlit versions
    else:
        st.markdown(
            content,
            unsafe_allow_html=unsafe_allow_html
        )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "Data"
    / "customer_churn_business_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "churn_model.pkl"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       FORCE LIGHT MODE
       ===================================================== */

    html {
        color-scheme: light !important;
    }

    body {
        color-scheme: light !important;
    }

    .stApp {
        background-color: #F3FAFA !important;
        color: #073B4C !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #F3FAFA !important;
    }

    [data-testid="stMain"] {
        background-color: #F3FAFA !important;
    }


    /* =====================================================
       MAIN CONTAINER
       ===================================================== */

    .block-container {
        padding-top: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }


    /* =====================================================
       FONT
       ===================================================== */

    * {
        font-family:
            "Inter",
            "Segoe UI",
            Arial,
            sans-serif;
    }


   /* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background-color: #006D6D !important;
    min-width: 255px !important;
    max-width: 255px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.2rem !important;
    padding-left: 13px !important;
    padding-right: 13px !important;
}


/* Sidebar title */

.sidebar-title {
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin-top: 0px !important;
    margin-bottom: 3px;
}

.sidebar-subtitle {
    color: #C8EEEE;
    font-size: 11px;
    margin-bottom: 20px;
}


/* Navigation heading */

.sidebar-navigation {
    color: white;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}


    /* =====================================================
       OPTION MENU
       ===================================================== */

    [data-testid="stSidebar"] .nav-link {
        color: #E9FFFF !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        border-radius: 7px !important;
        margin-top: 3px !important;
        margin-bottom: 3px !important;
        padding: 8px 10px !important;
    }


    [data-testid="stSidebar"] .nav-link:hover {
        background-color: rgba(255,255,255,0.12) !important;
        color: white !important;
    }


    [data-testid="stSidebar"] .nav-link-selected {
        background-color: #0A9292 !important;
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }


    [data-testid="stSidebar"] .nav-link i {
        color: white !important;
        font-size: 13px !important;
    }


    /* =====================================================
       SIDEBAR INFO
       ===================================================== */

        .sidebar-info {
        background-color: #087F8F;
        color: white;
        padding: 14px;
        border-radius: 9px;
        margin-top: 20px;
    }
    
    .sidebar-tech {
        font-size: 11px;
        font-weight: 600;
        line-height: 1.6;
    }
    
    .sidebar-created {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.18);
        color: #C8EEEE;
        font-size: 10px;
        line-height: 1.5;
    }
    
    .sidebar-created strong {
        color: white;
        font-size: 11px;
    }


    /* =====================================================
       PAGE TITLES
       ===================================================== */

    .main-title {
        color: #073B4C;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 3px;
    }


    .subtitle {
        color: #52717A;
        font-size: 12px;
        margin-bottom: 15px;
    }


    .section-title {
        color: #073B4C;
        font-size: 18px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 8px;
    }


    /* =====================================================
       KPI CARDS
       ===================================================== */

    .kpi-card {
        border-radius: 9px;
        padding: 13px 15px;
        min-height: 82px;
        border: 1px solid #DDEEEE;
        box-shadow: 0 2px 7px rgba(0,0,0,0.035);
    }


    .kpi-blue {
        background-color: #E4F6FA;
    }


    .kpi-red {
        background-color: #FFE8E8;
    }


    .kpi-yellow {
        background-color: #FFF4D8;
    }


    .kpi-green {
        background-color: #E2F7EF;
    }


    .kpi-label {
        color: #52717A;
        font-size: 11px;
        margin-bottom: 5px;
    }


    .kpi-value {
        color: #073B4C;
        font-size: 24px;
        font-weight: 700;
    }


    /* =====================================================
       CONTENT CARD
       ===================================================== */

    .content-card {
        background-color: white;
        border: 1px solid #DDEEEE;
        border-radius: 9px;
        padding: 14px;
        box-shadow: 0 2px 7px rgba(0,0,0,0.035);
    }


    /* =====================================================
       INSIGHT
       ===================================================== */

    .insight-box {
        background-color: #E2F7F5;
        border-left: 4px solid #00A6A6;
        border-radius: 8px;
        padding: 11px 14px;
        color: #245B61;
        font-size: 12px;
        margin-top: 10px;
    }


    /* =====================================================
       RISK BOXES
       ===================================================== */

    .risk-high {
        background-color: #FFE5E5;
        color: #C62828;
        padding: 13px;
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid #FFD0D0;
    }


    .risk-medium {
        background-color: #FFF4D6;
        color: #A66A00;
        padding: 13px;
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid #F6E2AC;
    }


    .risk-low {
        background-color: #E3F7EF;
        color: #087F5B;
        padding: 13px;
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid #C8EBDD;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        background-color: #008C8C !important;
        color: white !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        padding: 9px 18px !important;
    }


    .stButton > button:hover {
        background-color: #006D6D !important;
        color: white !important;
    }


    /* =====================================================
       INPUTS - FORCE LIGHT
       ===================================================== */

    input,
    textarea {
        color-scheme: light !important;
        background-color: #FFFFFF !important;
        color: #073B4C !important;
    }


    div[data-baseweb="input"] {
        border-radius: 7px;
        background-color: white !important;
    }


    div[data-baseweb="select"] {
        border-radius: 7px;
    }


    /* =====================================================
       PLOTLY
       ===================================================== */

    .js-plotly-plot,
    .plot-container {
        background-color: #FFFFFF !important;
    }


    /* =====================================================
       DATAFRAME
       ===================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }


    /* =====================================================
       STREAMLIT METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 9px;
        padding: 12px;
        border: 1px solid #DDEEEE;
    }


    [data-testid="stMetricLabel"] {
        color: #52717A !important;
        font-size: 11px !important;
    }


    [data-testid="stMetricValue"] {
        color: #073B4C !important;
        font-size: 25px !important;
    }


    /* =====================================================
       HIDE STREAMLIT BRANDING
       ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    return data


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    trained_model = joblib.load(MODEL_PATH)

    return trained_model


df = load_data()

model = load_model()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

categorical_cols = (
    df.select_dtypes(include=["object"])
    .columns
)

numeric_cols = (
    df.select_dtypes(
        include=["int64", "float64"]
    )
    .columns
)


for col in categorical_cols:

    if df[col].isnull().any():

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )


for col in numeric_cols:

    if df[col].isnull().any():

        df[col] = df[col].fillna(
            df[col].median()
        )


# =========================================================
# COMMON DATA
# =========================================================

health_df = df.copy()


# =========================================================
# CUSTOMER HEALTH SCORE
# =========================================================

health_df["login_score"] = (
    health_df["monthly_logins"]
    .rank(pct=True)
    * 100
)


health_df["tenure_score"] = (
    health_df["tenure_months"]
    .rank(pct=True)
    * 100
)


health_df["csat_health_score"] = (
    health_df["csat_score"]
    .rank(pct=True)
    * 100
)


health_df["payment_score"] = (
    (
        1
        -
        health_df["payment_failures"]
        .rank(pct=True)
    )
    * 100
)


health_df["activity_score"] = (
    (
        1
        -
        health_df["last_login_days_ago"]
        .rank(pct=True)
    )
    * 100
)


health_df["health_score"] = (

    health_df["login_score"] * 0.30

    +

    health_df["tenure_score"] * 0.15

    +

    health_df["csat_health_score"] * 0.25

    +

    health_df["payment_score"] * 0.15

    +

    health_df["activity_score"] * 0.15
)


health_df["health_score"] = (
    health_df["health_score"]
    .round(2)
)


health_df["health_status"] = pd.cut(

    health_df["health_score"],

    bins=[
        0,
        41,
        59,
        100
    ],

    labels=[
        "Critical",
        "At Risk",
        "Healthy"
    ],

    include_lowest=True
)


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

segment_cols = [

    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "tenure_months",
    "monthly_fee",
    "total_revenue",
    "support_tickets",
    "csat_score",
    "payment_failures"
]


scaler = StandardScaler()

X_segment = scaler.fit_transform(
    df[segment_cols]
)


kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)


health_df["cluster"] = (
    kmeans.fit_predict(X_segment)
)


cluster_names = {

    0:
        "Lower-Value / Newer Customers",

    1:
        "High-Value / Long-Term Customers"
}


health_df["customer_group"] = (
    health_df["cluster"]
    .map(cluster_names)
)


# =========================================================
# MODEL DATA
# =========================================================

X_all = df.drop(
    [
        "churn",
        "customer_id"
    ],
    axis=1
)


# =========================================================
# MODEL PREDICTION
# =========================================================

health_df["churn_probability"] = (

    model
    .predict_proba(X_all)[:, 1]
)


# =========================================================
# FINAL THRESHOLD
# =========================================================

final_threshold = 0.20


health_df["predicted_churn"] = (

    health_df["churn_probability"]
    >= final_threshold

).astype(int)


# =========================================================
# REVENUE AT RISK
# =========================================================

health_df["revenue_at_risk"] = (

    health_df["monthly_fee"]

    *

    health_df["churn_probability"]
)


# =========================================================
# RETENTION PRIORITY
# =========================================================

fee_median = (
    health_df["monthly_fee"]
    .median()
)


health_df["retention_priority"] = "Low"


health_df.loc[
    (
        health_df["churn_probability"]
        >= 0.20
    )
    &
    (
        health_df["monthly_fee"]
        >= fee_median
    ),
    "retention_priority"
] = "High"


health_df.loc[
    (
        health_df["churn_probability"]
        >= 0.20
    )
    &
    (
        health_df["monthly_fee"]
        < fee_median
    ),
    "retention_priority"
] = "Medium"


# =========================================================
# RETENTION RECOMMENDATIONS
# =========================================================

health_df[
    "retention_recommendation"
] = "Regular engagement"


health_df.loc[
    health_df["payment_failures"] > 0,
    "retention_recommendation"
] = "Payment support"


health_df.loc[
    (
        health_df["csat_score"] < 3
    )
    &
    (
        health_df["payment_failures"] == 0
    ),
    "retention_recommendation"
] = "Customer support follow-up"


health_df.loc[
    (
        health_df["monthly_logins"]
        <
        health_df["monthly_logins"].median()
    )
    &
    (
        health_df["payment_failures"] == 0
    )
    &
    (
        health_df["csat_score"] >= 3
    ),
    "retention_recommendation"
] = "Re-engagement campaign"


# =========================================================
# PLOTLY STYLE
# =========================================================

def style_chart(fig):

    fig.update_layout(

        template="plotly_white",

        paper_bgcolor="#FFFFFF",

        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Inter, Arial, sans-serif",
            color="#073B4C"
        ),

        title_font=dict(
            family="Inter, Arial, sans-serif",
            size=14,
            color="#073B4C"
        ),

        xaxis=dict(
            color="#52717A",
            gridcolor="#E5EEEE",
            zerolinecolor="#D5E5E5"
        ),

        yaxis=dict(
            color="#52717A",
            gridcolor="#E5EEEE",
            zerolinecolor="#D5E5E5"
        ),

        legend=dict(
            font=dict(
                color="#073B4C",
                size=10
            ),
            bgcolor="rgba(255,255,255,0)"
        ),

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-title">
            Customer Churn
        </div>

        <div class="sidebar-subtitle">
            Retention Intelligence
        </div>

        <div class="sidebar-navigation">
            Navigation
        </div>
        """
    )


    page = option_menu(

        menu_title=None,

        options=[
            "Overview",
            "Visualization",
            "Customer Segmentation",
            "Customer Health",
            "Churn Prediction",
            "Revenue at Risk",
            "Retention Priority",
            "Recommendations",
            "Customer Search",
            "About"
        ],

        icons=[
            "house",
            "bar-chart-line",
            "people",
            "heart-pulse",
            "robot",
            "cash-stack",
            "bullseye",
            "lightbulb",
            "search",
            "info-circle"
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "0",
                "margin": "0",
                "background-color": "#006D6D"
            },

            "icon": {
                "color": "white",
                "font-size": "13px"
            },

            "nav-link": {
                "font-size": "12px",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "8px 10px",
                "border-radius": "7px",
                "color": "#E9FFFF"
            },

            "nav-link-selected": {
                "background-color": "#0A9292",
                "color": "white",
                "font-weight": "700"
            }
        }
    )


    render_html(
    """
    <div class="sidebar-divider"></div>

    <div class="sidebar-info">

        <div class="sidebar-tech">
            Built with Python,<br>
            Machine Learning<br>
            and Streamlit
        </div>

        <div class="sidebar-created">
            Created by<br>
            <strong>Aditi Kumawat</strong>
        </div>

    </div>
    """
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    render_html(
        """
        <div class="main-title">
            Customer Churn & Retention Intelligence
        </div>

        <div class="subtitle">
            Understand your customers.
            Predict churn. Take action.
        </div>
        """
    )


    total_customers = len(df)


    churn_rate = (
        df["churn"].mean()
        * 100
    )


    at_risk = (
        health_df["health_status"]
        == "At Risk"
    ).sum()


    total_risk = (
        health_df["revenue_at_risk"]
        .sum()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        render_html(
            f"""
            <div class="kpi-card kpi-blue">

                <div class="kpi-label">
                    Total Customers
                </div>

                <div class="kpi-value">
                    {total_customers:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        render_html(
            f"""
            <div class="kpi-card kpi-red">

                <div class="kpi-label">
                    Churn Rate
                </div>

                <div class="kpi-value">
                    {churn_rate:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        render_html(
            f"""
            <div class="kpi-card kpi-yellow">

                <div class="kpi-label">
                    At-Risk Customers
                </div>

                <div class="kpi-value">
                    {at_risk:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        render_html(
            f"""
            <div class="kpi-card kpi-blue">

                <div class="kpi-label">
                    Revenue at Risk
                </div>

                <div class="kpi-value">
                    ₹{total_risk:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    render_html(
        """
        <div class="section-title">
            Customer Health & Churn Overview
        </div>
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        health_distribution = (

            health_df[
                "health_status"
            ]
            .value_counts()
            .reset_index()
        )


        health_distribution.columns = [
            "Health Status",
            "Customers"
        ]


        fig = px.pie(
            health_distribution,
            names="Health Status",
            values="Customers",
            hole=0.55,
            title="Customer Health Distribution"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        segment_churn = (

            df
            .groupby("customer_segment")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.bar(
            segment_churn,
            x="customer_segment",
            y="churn",
            labels={
                "customer_segment":
                    "Customer Segment",

                "churn":
                    "Churn Rate (%)"
            },
            title="Churn Rate by Customer Segment"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    render_html(
        f"""
        <div class="insight-box">

            <b>Insight:</b>

            Overall churn rate is
            {churn_rate:.2f}%.
            {at_risk:,} customers are currently
            classified as At Risk.

        </div>
        """
    )


# =========================================================
# VISUALIZATION
# =========================================================

elif page == "Visualization":

    render_html(
        """
        <div class="main-title">
            Exploratory Data Analysis
        </div>

        <div class="subtitle">
            Explore customer behavior and identify
            churn-related patterns.
        </div>
        """
    )


    visualization = st.selectbox(

        "Select Analysis",

        [
            "Churn vs CSAT",
            "Churn by Contract Type",
            "Churn by Customer Segment",
            "Churn vs Monthly Logins",
            "Churn vs Tenure",
            "Churn vs Payment Failures",
            "Churn vs Support Tickets"
        ]
    )


    # -----------------------------------------------------
    # CSAT
    # -----------------------------------------------------

    if visualization == "Churn vs CSAT":

        temp = (

            df
            .groupby("csat_score")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.line(

            temp,

            x="csat_score",

            y="churn",

            markers=True,

            labels={
                "csat_score":
                    "CSAT Score",

                "churn":
                    "Churn Rate (%)"
            },

            title="CSAT Score vs Churn"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    elif visualization == "Churn by Contract Type":

        temp = (

            df
            .groupby("contract_type")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.bar(

            temp,

            x="contract_type",

            y="churn",

            labels={
                "contract_type":
                    "Contract Type",

                "churn":
                    "Churn Rate (%)"
            },

            title="Churn Rate by Contract Type"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # CUSTOMER SEGMENT
    # -----------------------------------------------------

    elif visualization == "Churn by Customer Segment":

        temp = (

            df
            .groupby("customer_segment")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.bar(

            temp,

            x="customer_segment",

            y="churn",

            labels={
                "customer_segment":
                    "Customer Segment",

                "churn":
                    "Churn Rate (%)"
            },

            title="Churn Rate by Customer Segment"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # MONTHLY LOGINS
    # -----------------------------------------------------

    elif visualization == "Churn vs Monthly Logins":

        fig = px.box(

            df,

            x="churn",

            y="monthly_logins",

            points=False,

            labels={
                "churn":
                    "Churn",

                "monthly_logins":
                    "Monthly Logins"
            },

            title="Monthly Logins vs Churn"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # TENURE
    # -----------------------------------------------------

    elif visualization == "Churn vs Tenure":

        fig = px.box(

            df,

            x="churn",

            y="tenure_months",

            points=False,

            labels={
                "churn":
                    "Churn",

                "tenure_months":
                    "Tenure (Months)"
            },

            title="Tenure vs Churn"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PAYMENT FAILURES
    # -----------------------------------------------------

    elif visualization == "Churn vs Payment Failures":

        temp = (

            df
            .groupby("payment_failures")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.bar(

            temp,

            x="payment_failures",

            y="churn",

            labels={
                "payment_failures":
                    "Payment Failures",

                "churn":
                    "Churn Rate (%)"
            },

            title="Payment Failures vs Churn"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SUPPORT TICKETS
    # -----------------------------------------------------

    elif visualization == "Churn vs Support Tickets":

        temp = (

            df
            .groupby("support_tickets")
            ["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )


        fig = px.bar(

            temp,

            x="support_tickets",

            y="churn",

            labels={
                "support_tickets":
                    "Support Tickets",

                "churn":
                    "Churn Rate (%)"
            },

            title="Support Tickets vs Churn"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

elif page == "Customer Segmentation":

    render_html(
        """
        <div class="main-title">
            Customer Segmentation
        </div>

        <div class="subtitle">
            Group customers based on behavior,
            value and customer characteristics.
        </div>
        """
    )


    lower_count = (

        health_df["customer_group"]
        ==
        "Lower-Value / Newer Customers"

    ).sum()


    high_count = (

        health_df["customer_group"]
        ==
        "High-Value / Long-Term Customers"

    ).sum()


    col1, col2 = st.columns(2)


    with col1:

        render_html(
            f"""
            <div class="kpi-card kpi-blue">

                <div class="kpi-label">
                    Lower-Value / Newer
                </div>

                <div class="kpi-value">
                    {lower_count:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        render_html(
            f"""
            <div class="kpi-card kpi-green">

                <div class="kpi-label">
                    High-Value / Long-Term
                </div>

                <div class="kpi-value">
                    {high_count:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    pca = PCA(
        n_components=2
    )


    X_pca = pca.fit_transform(
        X_segment
    )


    pca_df = pd.DataFrame({

        "PCA 1":
            X_pca[:, 0],

        "PCA 2":
            X_pca[:, 1],

        "Customer Group":
            health_df["customer_group"]
    })


    fig = px.scatter(

        pca_df,

        x="PCA 1",

        y="PCA 2",

        color="Customer Group",

        title="Customer Segments - PCA Visualization"
    )


    fig = style_chart(fig)


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    segment_summary = (

        health_df

        .groupby("customer_group")

        .agg(

            Customers=(
                "customer_group",
                "count"
            ),

            Churn_Rate=(
                "churn",
                "mean"
            ),

            Avg_Health_Score=(
                "health_score",
                "mean"
            ),

            Avg_Monthly_Fee=(
                "monthly_fee",
                "mean"
            ),

            Avg_Total_Revenue=(
                "total_revenue",
                "mean"
            )
        )

        .reset_index()
    )


    segment_summary[
        "Churn_Rate"
    ] = (
        segment_summary[
            "Churn_Rate"
        ] * 100
    )


    st.dataframe(
        segment_summary.round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CUSTOMER HEALTH
# =========================================================

elif page == "Customer Health":

    render_html(
        """
        <div class="main-title">
            Customer Health Analysis
        </div>

        <div class="subtitle">
            Identify at-risk and critical customers
            based on health score.
        </div>
        """
    )


    critical = (

        health_df["health_status"]
        == "Critical"

    ).sum()


    at_risk = (

        health_df["health_status"]
        == "At Risk"

    ).sum()


    healthy = (

        health_df["health_status"]
        == "Healthy"

    ).sum()


    col1, col2, col3 = st.columns(3)


    with col1:

        render_html(
            f"""
            <div class="kpi-card kpi-red">

                <div class="kpi-label">
                    Critical
                </div>

                <div class="kpi-value">
                    {critical:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        render_html(
            f"""
            <div class="kpi-card kpi-yellow">

                <div class="kpi-label">
                    At Risk
                </div>

                <div class="kpi-value">
                    {at_risk:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        render_html(
            f"""
            <div class="kpi-card kpi-green">

                <div class="kpi-label">
                    Healthy
                </div>

                <div class="kpi-value">
                    {healthy:,}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    col1, col2 = st.columns(2)


    with col1:

        fig = px.histogram(

            health_df,

            x="health_score",

            nbins=25,

            title="Health Score Distribution"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        temp = (

            health_df[
                "health_status"
            ]

            .value_counts()

            .reset_index()
        )


        temp.columns = [
            "Health Status",
            "Customers"
        ]


        fig = px.pie(

            temp,

            names="Health Status",

            values="Customers",

            hole=0.50,

            title="Customers by Health Category"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    render_html(
        f"""
        <div class="insight-box">

            <b>Insight:</b>

            {at_risk / len(df) * 100:.2f}%
            of customers are classified as At Risk.

        </div>
        """
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "Churn Prediction":

    render_html(
        """
        <div class="main-title">
            Predict Customer Churn
        </div>

        <div class="subtitle">
            Enter customer details to predict
            churn probability.
        </div>
        """
    )


    render_html(
        """
        <div class="insight-box">

            <b>Note:</b><br><br>

            The model was trained using all available
            customer features. Fields not shown in this
            form use typical dataset values to keep
            the prediction interface simple.

        </div>
        """
    )


    # -----------------------------------------------------
    # MODEL COLUMNS
    # -----------------------------------------------------

    numeric_model_cols = [

        col

        for col in X_all.columns

        if pd.api.types.is_numeric_dtype(
            X_all[col]
        )
    ]


    categorical_model_cols = [

        col

        for col in X_all.columns

        if not pd.api.types.is_numeric_dtype(
            X_all[col]
        )
    ]


    input_data = {}


    # Default numerical values

    for col in numeric_model_cols:

        input_data[col] = float(
            df[col].median()
        )


    # Default categorical values

    for col in categorical_model_cols:

        input_data[col] = (
            df[col].mode()[0]
        )


    # =====================================================
    # USAGE & ENGAGEMENT
    # =====================================================

    render_html(
        """
        <div class="section-title">
            Usage & Engagement
        </div>
        """
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data[
            "monthly_logins"
        ] = st.number_input(

            "Monthly Logins",

            min_value=0,

            value=int(
                df["monthly_logins"].median()
            )
        )


        input_data[
            "weekly_active_days"
        ] = st.number_input(

            "Weekly Active Days",

            min_value=0,

            max_value=7,

            value=int(
                df["weekly_active_days"].median()
            )
        )


        input_data[
            "avg_session_time"
        ] = st.number_input(

            "Average Session Time",

            min_value=0.0,

            value=float(
                df["avg_session_time"].median()
            )
        )


    with col2:

        input_data[
            "features_used"
        ] = st.number_input(

            "Features Used",

            min_value=0,

            value=int(
                df["features_used"].median()
            )
        )


        input_data[
            "tenure_months"
        ] = st.number_input(

            "Tenure (Months)",

            min_value=0,

            value=int(
                df["tenure_months"].median()
            )
        )


        input_data[
            "last_login_days_ago"
        ] = st.number_input(

            "Last Login (Days Ago)",

            min_value=0,

            value=int(
                df["last_login_days_ago"].median()
            )
        )


    with col3:

        input_data[
            "usage_growth_rate"
        ] = st.number_input(

            "Usage Growth Rate",

            value=float(
                df["usage_growth_rate"].median()
            )
        )


        input_data[
            "monthly_fee"
        ] = st.number_input(

            "Monthly Fee (₹)",

            min_value=0.0,

            value=float(
                df["monthly_fee"].median()
            )
        )


        input_data[
            "support_tickets"
        ] = st.number_input(

            "Support Tickets",

            min_value=0,

            value=int(
                df["support_tickets"].median()
            )
        )


    # =====================================================
    # PAYMENT & CUSTOMER EXPERIENCE
    # =====================================================

    render_html(
        """
        <div class="section-title">
            Payment & Customer Experience
        </div>
        """
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        input_data[
            "payment_failures"
        ] = st.number_input(

            "Payment Failures",

            min_value=0,

            value=int(
                df["payment_failures"].median()
            )
        )


        input_data[
            "csat_score"
        ] = st.number_input(

            "CSAT Score",

            min_value=1.0,

            max_value=5.0,

            value=float(
                df["csat_score"].median()
            )
        )


    with col2:

        input_data[
            "customer_segment"
        ] = st.selectbox(

            "Customer Segment",

            sorted(
                df[
                    "customer_segment"
                ]
                .dropna()
                .unique()
            )
        )


        input_data[
            "contract_type"
        ] = st.selectbox(

            "Contract Type",

            sorted(
                df[
                    "contract_type"
                ]
                .dropna()
                .unique()
            )
        )


    with col3:

        input_data[
            "gender"
        ] = st.selectbox(

            "Gender",

            sorted(
                df[
                    "gender"
                ]
                .dropna()
                .unique()
            )
        )


        input_data[
            "country"
        ] = st.selectbox(

            "Country",

            sorted(
                df[
                    "country"
                ]
                .dropna()
                .unique()
            )
        )


    st.markdown("")


    predict_button = st.button(
        "Predict Churn",
        use_container_width=True
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if predict_button:

        prediction_df = pd.DataFrame(
            [input_data]
        )


        probability = (

            model
            .predict_proba(
                prediction_df
            )[0, 1]
        )


        predicted_churn = int(
            probability >= final_threshold
        )


        # -------------------------------------------------
        # HEALTH SCORE FUNCTION
        # -------------------------------------------------

        def percentile_score(
            value,
            reference,
            reverse=False
        ):

            score = (
                (
                    reference <= value
                ).mean()
                * 100
            )


            if reverse:

                score = 100 - score


            return score


        login_score = percentile_score(

            input_data[
                "monthly_logins"
            ],

            df[
                "monthly_logins"
            ]
        )


        tenure_score = percentile_score(

            input_data[
                "tenure_months"
            ],

            df[
                "tenure_months"
            ]
        )


        csat_score = percentile_score(

            input_data[
                "csat_score"
            ],

            df[
                "csat_score"
            ]
        )


        payment_score = percentile_score(

            input_data[
                "payment_failures"
            ],

            df[
                "payment_failures"
            ],

            reverse=True
        )


        activity_score = percentile_score(

            input_data[
                "last_login_days_ago"
            ],

            df[
                "last_login_days_ago"
            ],

            reverse=True
        )


        customer_health = (

            login_score * 0.30

            +

            tenure_score * 0.15

            +

            csat_score * 0.25

            +

            payment_score * 0.15

            +

            activity_score * 0.15
        )


        # -------------------------------------------------
        # HEALTH STATUS
        # -------------------------------------------------

        if customer_health <= 41:

            health_status = "Critical"

        elif customer_health <= 59:

            health_status = "At Risk"

        else:

            health_status = "Healthy"


        # -------------------------------------------------
        # REVENUE RISK
        # -------------------------------------------------

        revenue_risk = (

            input_data[
                "monthly_fee"
            ]

            *

            probability
        )


        # -------------------------------------------------
        # PRIORITY
        # -------------------------------------------------

        if (

            probability >= 0.20

            and

            input_data[
                "monthly_fee"
            ] >= fee_median

        ):

            priority = "High"


        elif probability >= 0.20:

            priority = "Medium"


        else:

            priority = "Low"


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        if (
            input_data[
                "payment_failures"
            ] > 0
        ):

            recommendation = (
                "Payment support"
            )


        elif (
            input_data[
                "csat_score"
            ] < 3
        ):

            recommendation = (
                "Customer support follow-up"
            )


        elif (
            input_data[
                "monthly_logins"
            ]
            <
            df[
                "monthly_logins"
            ].median()
        ):

            recommendation = (
                "Re-engagement campaign"
            )


        else:

            recommendation = (
                "Regular engagement"
            )


        # -------------------------------------------------
        # RESULT TITLE
        # -------------------------------------------------

        render_html(
            """
            <div class="section-title">
                Prediction Result
            </div>
            """
        )


        col1, col2 = st.columns(2)


        with col1:

            render_html(
                f"""
                <div class="kpi-card kpi-red">

                    <div class="kpi-label">
                        Churn Probability
                    </div>

                    <div class="kpi-value">
                        {probability * 100:.1f}%
                    </div>

                </div>
                """
            )


            st.markdown("")


            render_html(
                f"""
                <div class="kpi-card kpi-blue">

                    <div class="kpi-label">
                        Customer Health Score
                    </div>

                    <div class="kpi-value">
                        {customer_health:.1f}
                    </div>

                </div>
                """
            )


            st.markdown("")


            render_html(
                f"""
                <div class="kpi-card kpi-yellow">

                    <div class="kpi-label">
                        Revenue at Risk
                    </div>

                    <div class="kpi-value">
                        ₹{revenue_risk:.2f}
                    </div>

                </div>
                """
            )


        with col2:

            if predicted_churn:

                render_html(
                    """
                    <div class="risk-high">

                        HIGH CHURN RISK

                        <br>

                        <span style="
                            font-size:11px;
                            font-weight:400;
                        ">

                        This customer is predicted
                        to be at risk of churn.

                        </span>

                    </div>
                    """
                )

            else:

                render_html(
                    """
                    <div class="risk-low">

                        LOWER CHURN RISK

                        <br>

                        <span style="
                            font-size:11px;
                            font-weight:400;
                        ">

                        This customer is not
                        currently predicted to churn.

                        </span>

                    </div>
                    """
                )


            st.markdown("")


            st.write(
                f"**Health Status:** {health_status}"
            )


            st.write(
                f"**Retention Priority:** {priority}"
            )


            st.write(
                f"**Recommendation:** {recommendation}"
            )


# =========================================================
# REVENUE AT RISK
# =========================================================

elif page == "Revenue at Risk":

    render_html(
        """
        <div class="main-title">
            Revenue at Risk
        </div>

        <div class="subtitle">
            Estimate the monthly revenue exposure
            associated with churn risk.
        </div>
        """
    )


    total_risk = (
        health_df[
            "revenue_at_risk"
        ].sum()
    )


    avg_risk = (
        health_df[
            "revenue_at_risk"
        ].mean()
    )


    at_risk_customers = (
        health_df[
            "churn_probability"
        ]
        >= 0.20
    ).sum()


    col1, col2, col3 = st.columns(3)


    with col1:

        render_html(
            f"""
            <div class="kpi-card kpi-blue">

                <div class="kpi-label">
                    Total Revenue at Risk
                </div>

                <div class="kpi-value">
                    ₹{total_risk:,.2f}
                </div>

            </div>
            """
        )


    with col2:

        render_html(
            f"""
            <div class="kpi-card kpi-green">

                <div class="kpi-label">
                    Average Revenue at Risk
                </div>

                <div class="kpi-value">
                    ₹{avg_risk:,.2f}
                </div>

            </div>
            """
        )


    with col3:

        render_html(
            f"""
            <div class="kpi-card kpi-red">

                <div class="kpi-label">
                    At-Risk Customers
                </div>

                <div class="kpi-value">
                    {at_risk_customers:,}
                </div>

            </div>
            """
        )


    st.markdown("")


    col1, col2 = st.columns(2)


    with col1:

        top10 = (

            health_df

            .nlargest(
                10,
                "revenue_at_risk"
            )

            .sort_values(
                "revenue_at_risk"
            )
        )


        fig = px.bar(

            top10,

            x="revenue_at_risk",

            y="customer_id",

            orientation="h",

            title=
                "Top 10 Customers by Revenue at Risk"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        segment_risk = (

            health_df

            .groupby(
                "customer_group"
            )["revenue_at_risk"]

            .sum()

            .reset_index()
        )


        fig = px.bar(

            segment_risk,

            x="customer_group",

            y="revenue_at_risk",

            title=
                "Revenue at Risk by Customer Group"
        )


        fig = style_chart(fig)


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.dataframe(

        health_df[
            [
                "customer_id",
                "monthly_fee",
                "churn_probability",
                "revenue_at_risk"
            ]
        ]

        .sort_values(
            "revenue_at_risk",
            ascending=False
        )

        .head(20),

        use_container_width=True,

        hide_index=True
    )


# =========================================================
# RETENTION PRIORITY
# =========================================================

elif page == "Retention Priority":

    render_html(
        """
        <div class="main-title">
            Customer Retention Priority
        </div>

        <div class="subtitle">
            Identify and prioritize customers
            for retention efforts.
        </div>
        """
    )


    high = (

        health_df[
            "retention_priority"
        ]
        == "High"
    ).sum()


    medium = (

        health_df[
            "retention_priority"
        ]
        == "Medium"
    ).sum()


    low = (

        health_df[
            "retention_priority"
        ]
        == "Low"
    ).sum()


    col1, col2, col3 = st.columns(3)


    with col1:

        render_html(
            f"""
            <div class="kpi-card kpi-red">

                <div class="kpi-label">
                    High Priority
                </div>

                <div class="kpi-value">
                    {high:,}
                </div>

            </div>
            """
        )


    with col2:

        render_html(
            f"""
            <div class="kpi-card kpi-yellow">

                <div class="kpi-label">
                    Medium Priority
                </div>

                <div class="kpi-value">
                    {medium:,}
                </div>

            </div>
            """
        )


    with col3:

        render_html(
            f"""
            <div class="kpi-card kpi-green">

                <div class="kpi-label">
                    Low Priority
                </div>

                <div class="kpi-value">
                    {low:,}
                </div>

            </div>
            """
        )


    st.markdown("")


    priority_df = pd.DataFrame({

        "Priority": [
            "High",
            "Medium",
            "Low"
        ],

        "Customers": [
            high,
            medium,
            low
        ]
    })


    fig = px.pie(

        priority_df,

        names="Priority",

        values="Customers",

        hole=0.55,

        title="Customer Priority Distribution"
    )


    fig = style_chart(fig)


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    render_html(
        """
        <div class="section-title">
            Top Priority Customers
        </div>
        """
    )


    priority_table = (

        health_df[
            health_df[
                "retention_priority"
            ] == "High"
        ]

        [
            [
                "customer_id",
                "churn_probability",
                "health_score",
                "monthly_fee",
                "revenue_at_risk",
                "retention_priority"
            ]
        ]

        .sort_values(
            "revenue_at_risk",
            ascending=False
        )

        .head(20)
    )


    st.dataframe(
        priority_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "Recommendations":

    render_html(
        """
        <div class="main-title">
            Retention Recommendations
        </div>

        <div class="subtitle">
            Suggested actions based on customer
            risk factors.
        </div>
        """
    )


    recommendation_counts = (

        health_df[
            "retention_recommendation"
        ]

        .value_counts()

        .reset_index()
    )


    recommendation_counts.columns = [
        "Recommendation",
        "Customers"
    ]


    fig = px.bar(

        recommendation_counts,

        x="Recommendation",

        y="Customers",

        title=
            "Retention Recommendation Distribution"
    )


    fig = style_chart(fig)


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.dataframe(

        health_df[
            [
                "customer_id",
                "churn_probability",
                "health_score",
                "retention_priority",
                "retention_recommendation"
            ]
        ]

        .sort_values(
            "churn_probability",
            ascending=False
        )

        .head(50),

        use_container_width=True,

        hide_index=True
    )


# =========================================================
# CUSTOMER SEARCH
# =========================================================

elif page == "Customer Search":

    render_html(
        """
        <div class="main-title">
            Customer Search
        </div>

        <div class="subtitle">
            Search for a customer and view their
            churn risk and recommendation.
        </div>
        """
    )


    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="Example: CUST_00001"
    )


    if customer_id:

        result = (

            health_df[
                health_df[
                    "customer_id"
                ]
                .astype(str)
                .str.upper()
                ==
                customer_id.upper()
            ]
        )


        if len(result) == 0:

            st.error(
                "Customer ID not found."
            )


        else:

            customer = result.iloc[0]


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(

                    "Churn Probability",

                    f"""
                    {customer[
                        'churn_probability'
                    ] * 100:.1f}%
                    """
                )


            with col2:

                st.metric(

                    "Health Score",

                    f"""
                    {customer[
                        'health_score'
                    ]:.1f}
                    """
                )


            with col3:

                st.metric(

                    "Revenue at Risk",

                    f"""
                    ₹{customer[
                        'revenue_at_risk'
                    ]:.2f}
                    """
                )


            with col4:

                st.metric(

                    "Monthly Fee",

                    f"""
                    ₹{customer[
                        'monthly_fee'
                    ]:.2f}
                    """
                )


            render_html(
                """
                <div class="section-title">
                    Customer Details
                </div>
                """
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    f"**Customer ID:** "
                    f"{customer['customer_id']}"
                )


                st.write(
                    f"**Customer Segment:** "
                    f"{customer['customer_segment']}"
                )


                st.write(
                    f"**Contract Type:** "
                    f"{customer['contract_type']}"
                )


                st.write(
                    f"**Tenure:** "
                    f"{customer['tenure_months']} months"
                )


                st.write(
                    f"**CSAT:** "
                    f"{customer['csat_score']}"
                )


            with col2:

                st.write(
                    f"**Health Status:** "
                    f"{customer['health_status']}"
                )


                st.write(
                    f"**Priority:** "
                    f"{customer['retention_priority']}"
                )


                st.write(
                    f"**Customer Group:** "
                    f"{customer['customer_group']}"
                )


                st.write(
                    f"**Recommendation:** "
                    f"{customer['retention_recommendation']}"
                )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    render_html(
        """
        <div class="main-title">
            About This Project
        </div>

        <div class="subtitle">
            Customer Churn & Retention Intelligence
        </div>
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        render_html(
            """
            <div class="content-card">

                <h3>
                    Customer Churn & Retention Intelligence
                </h3>

                This project analyzes customer behavior
                to understand churn, predict churn risk,
                estimate revenue at risk and suggest
                retention actions.

                <br><br>

                <b>Dataset</b>

                <ul>

                    <li>
                        10,000 customers
                    </li>

                    <li>
                        Customer behavior and
                        business features
                    </li>

                    <li>
                        Churn target
                    </li>

                </ul>

            </div>
            """
        )


    with col2:

        render_html(
            """
            <div class="content-card">

                <h3>
                    Machine Learning
                </h3>

                <ul>

                    <li>
                        Logistic Regression
                    </li>

                    <li>
                        Random Forest
                    </li>

                    <li>
                        Gradient Boosting
                    </li>

                    <li>
                        GridSearchCV
                    </li>

                    <li>
                        Threshold tuning
                    </li>

                </ul>

                <b>Business Analysis</b>

                <ul>

                    <li>
                        Customer Health Score
                    </li>

                    <li>
                        Customer Segmentation
                    </li>

                    <li>
                        Revenue at Risk
                    </li>

                    <li>
                        Retention Priority
                    </li>

                    <li>
                        Retention Recommendations
                    </li>

                </ul>

            </div>
            """
        )


    render_html(
        """
        <div class="insight-box">

            <b>Project Goal</b>

            <br><br>

            Understand customer behavior,
            identify churn risk, estimate revenue
            exposure and support data-driven
            retention decisions.

        </div>
        """
    )