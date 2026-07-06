"""
Customer Churn Prediction — Streamlit App
==========================================
Loads a pre-trained Keras ANN model + preprocessing artifacts
(OneHotEncoder for Geography, LabelEncoder for Gender, StandardScaler)
and provides an interactive UI for live single-customer and batch predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import tensorflow as tf

# ----------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2.1rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .risk-high  { background: linear-gradient(135deg, #ff4b4b, #b30000); }
    .risk-med   { background: linear-gradient(135deg, #ffa53d, #b35c00); }
    .risk-low   { background: linear-gradient(135deg, #21c15e, #0a7a34); }
    
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    
    /* Sidebar Visibility Fix - High contrast text and background styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f3f6 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# EXPECTED FEATURE ORDER (matches training pipeline, target 'Exited' excluded)
# ----------------------------------------------------------------------
FEATURE_ORDER = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
]

REQUIRED_BATCH_COLUMNS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


# ----------------------------------------------------------------------
# ARTIFACT LOADING (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and preprocessing artifacts...")
def load_artifacts():
    errors = []
    
    model = None
    onehotencoder = None
    label_encoder_gender = None
    scaler = None
    
    try:
        model = tf.keras.models.load_model("model.h5")
    except Exception as e:
        errors.append(f"**model.h5**: {e}")
        
    try:
        with open("onehotencoder.pkl", "rb") as file:
            onehotencoder = pickle.load(file)
    except Exception as e:
        errors.append(f"**onehotencoder.pkl**: {e}")
        
    try:
        with open("label_encoder_gender.pkl", "rb") as file:
            label_encoder_gender = pickle.load(file)
    except Exception as e:
        errors.append(f"**label_encoder_gender.pkl**: {e}")
        
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
    except Exception as e:
        errors.append(f"**scaler.pkl**: {e}")
        
    return model, onehotencoder, label_encoder_gender, scaler, errors


model, onehotencoder, label_encoder_gender, scaler, load_errors = load_artifacts()

# Header (always shown)
st.markdown(
    """
    <div class="main-header">
    <h1>📉 Customer Churn Prediction</h1>
    <p>Estimate the likelihood that a bank customer will exit, using a trained ANN model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_errors:
    st.error(
        "⚠️ One or more required files could not be loaded. Please make sure "
        "`model.h5`, `onehotencoder.pkl`, `label_encoder_gender.pkl`, and "
        "`scaler.pkl` are in the same directory as this app."
    )
    with st.expander("Show error details"):
        for err in load_errors:
            st.markdown(f"- {err}")
    st.stop()


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def encode_geography(geography: str) -> dict:
    """One-hot encode Geography using the fitted encoder, aligned to expected columns."""
    try:
        encoded = onehotencoder.transform([[geography]])
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        cols = list(onehotencoder.get_feature_names_out(["Geography"]))
        row = dict(zip(cols, encoded[0]))
    except Exception:
        row = {}
        
    geo_cols = ["Geography_France", "Geography_Germany", "Geography_Spain"]
    result = {c: 0.0 for c in geo_cols}
    for c in geo_cols:
        if c in row:
            result[c] = float(row[c])
    if sum(result.values()) == 0 and f"Geography_{geography}" in geo_cols:
        result[f"Geography_{geography}"] = 1.0
    return result


def build_feature_row(credit_score, geography, gender, age, tenure, balance,
                      num_products, has_cr_card, is_active_member, estimated_salary):
    """Builds a single-row DataFrame in the exact column order the scaler/model expect."""
    gender_encoded = int(label_encoder_gender.transform([gender])[0])
    geo_encoded = encode_geography(geography)
    
    row = {
        "CreditScore": credit_score,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        **geo_encoded,
    }
    df = pd.DataFrame([row])[FEATURE_ORDER]
    return df


def predict_single(df_row: pd.DataFrame) -> float:
    """Scales the input row and returns the churn probability (0-1)."""
    scaled = scaler.transform(df_row)
    prob = model.predict(scaled, verbose=0)
    return float(np.ravel(prob)[0])


def risk_bucket(prob: float, threshold: float):
    if prob >= max(threshold, 0.6):
        return "High Risk", "risk-high", "🔴"
    elif prob >= threshold:
        return "Medium Risk", "risk-med", "🟠"
    else:
        return "Low Risk", "risk-low", "🟢"


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05, max_value=0.95, value=0.5, step=0.05,
        help="Probability above which a customer is classified as likely to churn.",
    )
    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        """
        This app uses an **Artificial Neural Network (ANN)** trained on 
        bank customer data to predict churn probability.
        
        **Inputs used by the model:**
        - Credit Score, Geography, Gender, Age
        - Tenure, Balance, Number of Products
        - Credit Card ownership, Active Membership
        - Estimated Salary
        """
    )
    st.divider()
    st.caption("Built with Streamlit • TensorFlow/Keras")


# ----------------------------------------------------------------------
# MAIN TABS
# ----------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🧍 Single Prediction", "📂 Batch Prediction (CSV)"])

# ========================================================================
# TAB 1 — SINGLE PREDICTION (LIVE UPDATING)
# ========================================================================
with tab_single:
    st.subheader("Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Personal Info**")
        gender = st.selectbox("Gender", options=list(label_encoder_gender.classes_))
        age = st.slider("Age", min_value=18, max_value=100, value=35)
        geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
        
    with col2:
        st.markdown("**Account Details**")
        tenure = st.slider("Tenure (years with bank)", min_value=0, max_value=10, value=3)
        num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)
        has_cr_card = st.radio("Has Credit Card?", options=["Yes", "No"], horizontal=True)
        is_active_member = st.radio("Is Active Member?", options=["Yes", "No"], horizontal=True)
        
    with col3:
        st.markdown("**Financial Info**")
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650, step=1
        )
        balance = st.number_input(
            "Account Balance", min_value=0.0, max_value=1_000_000.0,
            value=50000.0, step=100.0, format="%.2f"
        )
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, max_value=1_000_000.0,
            value=60000.0, step=100.0, format="%.2f"
        )
        
    # --- Live Prediction Output Block ---
    try:
        input_df = build_feature_row(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_products=num_products,
            has_cr_card=1 if has_cr_card == "Yes" else 0,
            is_active_member=1 if is_active_member == "Yes" else 0,
            estimated_salary=estimated_salary,
        )
        
        prob = predict_single(input_df)
        label, css_class, emoji = risk_bucket(prob, threshold)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div style="font-size:2rem;">{emoji} {label}</div>
                    <div style="font-size:1.1rem; margin-top:0.4rem;">
                        Churn Probability: {prob*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        with res_col2:
            st.metric("Live Churn Probability", f"{prob*100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))
            verdict = "likely to churn" if prob >= threshold else "likely to stay"
            st.write(f"Based on a threshold of **{threshold:.2f}**, this customer is **{verdict}**.")

        with st.expander("🔍 View processed input (model-ready)"):
            st.dataframe(input_df, use_container_width=True)
            
    except ValueError as ve:
        st.error(f"❌ Input processing error: {ve}")
    except KeyError as ke:
        st.error(f"❌ Missing column error: {ke}")
    except Exception as e:
        st.error("❌ Prediction error occurred.")
        with st.expander("Show technical details"):
            st.exception(e)

# ========================================================================
# TAB 2 — BATCH PREDICTION
# ========================================================================
with tab_batch:
    st.subheader("Upload a CSV for Batch Prediction")
    st.caption(
        "Required columns: "
        + ", ".join(f"`{c}`" for c in REQUIRED_BATCH_COLUMNS)
        + ". `Geography` should contain values France, Germany, or Spain. "
        "`HasCrCard`/`IsActiveMember` should be 0/1 or Yes/No."
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read the CSV file: {e}")
            batch_df = None
            
        if batch_df is not None:
            missing_cols = [c for c in REQUIRED_BATCH_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ The uploaded file is missing required columns: {missing_cols}")
            else:
                st.success(f"✅ File loaded successfully — {len(batch_df)} rows found.")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    progress_bar = st.progress(0, text="Starting predictions...")
                    results = []
                    error_rows = []
                    
                    total = len(batch_df)
                    for idx, row in batch_df.iterrows():
                        try:
                            def to_binary(v):
                                if isinstance(v, str):
                                    return 1 if v.strip().lower() in ("yes", "1", "true") else 0
                                return int(v)
                                
                            gender_val = str(row["Gender"]).strip()
                            if gender_val not in list(label_encoder_gender.classes_):
                                raise ValueError(f"Unknown Gender value '{gender_val}'")
                                
                            geography_val = str(row["Geography"]).strip()
                            if geography_val not in ["France", "Germany", "Spain"]:
                                raise ValueError(f"Unknown Geography value '{geography_val}'")
                                
                            feat_df = build_feature_row(
                                credit_score=float(row["CreditScore"]),
                                geography=geography_val,
                                gender=gender_val,
                                age=float(row["Age"]),
                                tenure=float(row["Tenure"]),
                                balance=float(row["Balance"]),
                                num_products=float(row["NumOfProducts"]),
                                has_cr_card=to_binary(row["HasCrCard"]),
                                is_active_member=to_binary(row["IsActiveMember"]),
                                estimated_salary=float(row["EstimatedSalary"]),
                            )
                            prob = predict_single(feat_df)
                            results.append(prob)
                        except Exception as row_err:
                            results.append(np.nan)
                            error_rows.append((idx, str(row_err)))
                            
                        progress_bar.progress(
                            (idx + 1) / total, text=f"Processing row {idx + 1} of {total}..."
                        )
                        
                    batch_df["Churn_Probability"] = results
                    batch_df["Prediction"] = np.where(
                        batch_df["Churn_Probability"] >= threshold, "Churn", "Stay"
                    )
                    batch_df.loc[batch_df["Churn_Probability"].isna(), "Prediction"] = "Error"
                    
                    progress_bar.empty()
                    st.success("✅ Batch prediction complete!")
                    
                    if error_rows:
                        with st.expander(f"⚠️ {len(error_rows)} row(s) had errors"):
                            for idx, msg in error_rows:
                                st.write(f"Row {idx}: {msg}")
                                
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_buffer = io.StringIO()
                    batch_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )