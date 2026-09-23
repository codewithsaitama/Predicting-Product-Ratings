import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Product Rating Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom E-Commerce & Review Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #78350F 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        margin-bottom: 10px;
    }

    .rating-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .rating-hero {
        font-size: 3.0rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #FBBF24;
        margin: 6px 0;
    }

    .sentiment-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Product_Rating_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("product_rating_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">E-Commerce & Customer Experience AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">⭐ Product Rating Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast customer star ratings (1.0 - 5.0 ⭐) based on perceived product quality, value for money, delivery speed, and customer support.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Product Experience Estimator", "📁 Batch Catalog Ratings (CSV)", "📊 Feature Importance & Diagnostics"])

# --- TAB 1: Product Experience Estimator ---
with tabs[0]:
    st.subheader("Customer Satisfaction & Service Dimensions")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_top = st.button("🌟 5-Star Flawless Experience", width="stretch")
    with p_cols[1]:
        load_low = st.button("⚠️ Poor Subpar Experience", width="stretch")
    with p_cols[2]:
        load_sample = st.button("📋 Sample Review", width="stretch")

    if load_top:
        st.session_state["qual"] = 10
        st.session_state["val"] = 9
        st.session_state["deliv"] = 10
        st.session_state["supp"] = 9
        st.session_state["ver"] = "Yes"
    elif load_low:
        st.session_state["qual"] = 2
        st.session_state["val"] = 2
        st.session_state["deliv"] = 3
        st.session_state["supp"] = 2
        st.session_state["ver"] = "No"
    elif load_sample:
        st.session_state["qual"] = 9
        st.session_state["val"] = 8
        st.session_state["deliv"] = 9
        st.session_state["supp"] = 8
        st.session_state["ver"] = "Yes"

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🛍️ Core Product Value")
        quality = st.slider(
            "Product Build & Material Quality (1 - 10)", 1, 10,
            value=int(st.session_state.get("qual", 9)), step=1,
            key="input_qual", help="Durability, finish, and operational performance."
        )
        value_money = st.slider(
            "Value for Money Rating (1 - 10)", 1, 10,
            value=int(st.session_state.get("val", 8)), step=1,
            key="input_val", help="Perceived benefit relative to purchase price paid."
        )
        verified_str = st.selectbox(
            "Verified Purchase Review?",
            ["Yes", "No"],
            index=["Yes", "No"].index(st.session_state.get("ver", "Yes")),
            key="input_ver"
        )
        verified_val = 1 if verified_str == "Yes" else 0

    with c_right:
        st.markdown("#### 🚚 Fulfillment & Support Services")
        delivery = st.slider(
            "Delivery & Packaging Rating (1 - 10)", 1, 10,
            value=int(st.session_state.get("deliv", 9)), step=1,
            key="input_deliv", help="Shipping promptness, tracking accuracy, and packaging protection."
        )
        support = st.slider(
            "Customer Support Responsiveness (1 - 10)", 1, 10,
            value=int(st.session_state.get("supp", 8)), step=1,
            key="input_supp", help="Post-purchase assistance, query resolution, and return flexibility."
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Calculate Estimated Customer Rating", type="primary", width="stretch")

    sample_df = pd.DataFrame([{
        "product_quality": quality,
        "value_for_money": value_money,
        "delivery_rating": delivery,
        "customer_support": support,
        "verified_purchase": verified_val
    }])

    pred_raw = float(model.predict(sample_df)[0])
    pred_rating = max(1.0, min(5.0, pred_raw))

    # Star visual
    stars_full = int(np.floor(pred_rating))
    stars_str = "⭐" * stars_full

    st.markdown("### 📋 Customer Review Rating Forecast")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        if pred_rating >= 4.5:
            sentiment_tier = "Excellent (Top-Rated)"
            color = "#10B981"
        elif pred_rating >= 4.0:
            sentiment_tier = "Good (Positive Sentiment)"
            color = "#F59E0B"
        elif pred_rating >= 3.0:
            sentiment_tier = "Mediocre (Needs Attention)"
            color = "#FB923C"
        else:
            sentiment_tier = "Critical (Negative Feedback)"
            color = "#EF4444"

        st.markdown(f"""
        <div class="rating-card">
            <span style="font-size: 2.4rem;">{stars_str}</span>
            <div style="color: {color}; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px;">
                {sentiment_tier}
            </div>
            <div class="rating-hero" style="color: {color};">
                {pred_rating:.2f} <span style="font-size: 1.4rem; color: #94A3B8;">/ 5.00</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Marketplace Star Rating</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float((pred_rating - 1.0) / 4.0))

    with r2:
        st.markdown("#### 💡 Satisfaction Breakdown & Insights")
        notes = []
        if quality >= 9:
            notes.append(("Superior Craftsmanship", "Quality score heavily anchors upper 5-star distribution.", "good"))
        elif quality <= 4:
            notes.append(("Quality Drag Concern", "Low build score is primary driver of review demotion.", "bad"))

        if value_money >= 8:
            notes.append(("Favorable Price-to-Value Ratio", "Consumers feel purchase justifies the investment.", "good"))

        if delivery <= 4:
            notes.append(("Fulfillment Friction", "Slow delivery or damaged packaging lowers net rating.", "bad"))

        if notes:
            for title, desc, tone in notes:
                if tone == "good":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("Service metrics are balanced across standard marketplace benchmarks.")

        st.markdown(f"""
        <div class="sentiment-box">
            <strong style="color: #FBBF24;">Merchant Optimization Action:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Showcase customer testimonials and promote as featured Amazon Choice / Best Seller.' if pred_rating >= 4.5 else ('Address fulfillment delays and optimize packaging to push rating into 4.5+ tier.' if pred_rating >= 4.0 else 'Initiate immediate product redesign and inspect supplier batch quality.')}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Features"):
        st.dataframe(sample_df, width="stretch")

# --- TAB 2: Batch Catalog Ratings ---
with tabs[1]:
    st.subheader("Batch Product Catalog Rating Simulation")
    st.write("Upload a product catalog CSV or evaluate against the 500-item baseline dataset.")

    csv_file = st.file_uploader("Upload Product Catalog CSV", type=["csv"], key="rating_csv")
    df_catalog = None

    if csv_file is not None:
        df_catalog = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_catalog)} product records from file.")
    else:
        sample_path = get_asset_path("data/product_ratings.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline product rating dataset (`data/product_ratings.csv`)", value=True):
                df_catalog = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_catalog)} records from baseline catalog dataset.")

    if df_catalog is not None:
        req_cols = ["product_quality", "value_for_money", "delivery_rating", "customer_support", "verified_purchase"]
        missing = [c for c in req_cols if c not in df_catalog.columns]
        if missing:
            st.error(f"Missing columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Catalog Rating Forecast", type="primary"):
                with st.spinner("Scoring product catalog..."):
                    preds = model.predict(df_catalog[req_cols])
                    preds = np.clip(preds, 1.0, 5.0)

                    res_df = df_catalog.copy()
                    res_df["Predicted_Rating"] = np.round(preds, 2)
                    res_df["Rating_Tier"] = [
                        "5-Star (≥4.8)" if r >= 4.8 else ("4-Star (4.0-4.79)" if r >= 4.0 else "Sub-4.0 (<4.0)")
                        for r in preds
                    ]

                    top_c = sum(preds >= 4.8)
                    risk_c = sum(preds < 4.0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Products", len(res_df))
                    m2.metric("Average Rating", f"{np.mean(preds):.2f} ⭐")
                    m3.metric("5-Star Champions", top_c)
                    m4.metric("Sub-4.0 At Risk", risk_c)

                    f_tier = st.selectbox("Filter by Rating Tier:", ["All", "5-Star (≥4.8)", "4-Star (4.0-4.79)", "Sub-4.0 (<4.0)"])
                    view = res_df
                    if f_tier != "All":
                        view = view[view["Rating_Tier"] == f_tier]

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Catalog as CSV",
                        data=csv_export,
                        file_name="product_catalog_rating_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Feature Importance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Rating Regressor Specifications
        - **Algorithm**: `RandomForestRegressor(n_estimators=200, random_state=42)`
        - **Continuous Target**: Star Rating ($1.00$ to $5.00$ ⭐)
        - **Evaluation Benchmark**:
            - **R² Score**: **0.8172**
            - **MAE (Mean Absolute Error)**: **0.12** stars
            - **RMSE**: ~0.22 stars
        - Precision alignment indicates highly stable rating forecast across varied delivery & support quality.
        """)

    with c2:
        img_path = get_asset_path("feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Importance in Driving Product Ratings", width="stretch")
        else:
            st.info("Feature importance image not found.")

st.caption("E-Commerce Customer Satisfaction & Review Analytics • Scikit-learn & Streamlit")
