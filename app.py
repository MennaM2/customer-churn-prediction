import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# ---------- Load data & model ----------
model = joblib.load("models/final_churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
scaler = joblib.load("models/scaler.pkl")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")
df = pd.read_csv("cleaned_churn_data.csv")   

explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------- Custom styling: gray buttons instead of orange/red ----------
st.markdown(
    """
    <style>
    div.stButton > button, button[kind="primary"] {
        background-color: #6c757d !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button:hover, button[kind="primary"]:hover {
        background-color: #565e64 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Customer Churn Dashboard")

# ---------- Filters ----------
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    contract_filter = st.selectbox("Contract", ["All"] + df["Contract"].unique().tolist())
with col_f2:
    internet_filter = st.selectbox("Internet Service", ["All"] + df["InternetService"].unique().tolist())
with col_f3:
    payment_filter = st.selectbox("Payment Method", ["All"] + df["PaymentMethod"].unique().tolist())

filtered_df = df.copy()
if contract_filter != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]
if internet_filter != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == internet_filter]
if payment_filter != "All":
    filtered_df = filtered_df[filtered_df["PaymentMethod"] == payment_filter]

# ---------- KPI Cards ----------
# NOTE: "Churn" is stored as 0/1 in cleaned_churn_data.csv (0 = No, 1 = Yes)
total_customers = len(filtered_df)
churn_rate = (filtered_df["Churn"] == 1).mean() * 100
avg_charges = filtered_df["MonthlyCharges"].mean()
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Churn Rate", f"{churn_rate:.1f}%")
k3.metric("Avg Monthly Charges", f"${avg_charges:.1f}")
k4.metric("Model ROC-AUC", f"{roc_auc:.3f}")

# ---------- Charts ----------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.subheader("Churn Rate by Contract Type")
    churn_by_contract = (
        filtered_df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == 1).mean() * 100)
        .reset_index(name="Churn Rate")
    )
    churn_by_contract["Churn Rate"] = churn_by_contract["Churn Rate"].round(1)

    if churn_by_contract.empty or churn_by_contract["Churn Rate"].sum() == 0:
        st.info("No churn data available for the current filter selection.")
    else:
        color_map = {
            "Month-to-month": "#e74c3c",
            "One year": "#f39c12",
            "Two year": "#2ecc71",
        }
        fig1 = px.bar(
            churn_by_contract,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            color_discrete_map=color_map,
            text="Churn Rate",
        )
        fig1.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            marker_line_width=0,
        )
        fig1.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10),
            yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        )
        st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("Customer Distribution")
    churn_labels = filtered_df["Churn"].map({0: "No", 1: "Yes"})
    fig2 = px.pie(
        names=churn_labels,
        hole=0.5,
        color=churn_labels,
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- Prediction Section ----------
st.subheader("Predict a New Customer")
st.caption("Fill in the customer details across the tabs below")

tab1, tab2, tab3 = st.tabs(["Customer Info", "Services", "Billing & Contract"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with col2:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col2:
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, float(tenure * monthly_charges))

st.write("")
predict_clicked = st.button("Calculate Churn Probability", use_container_width=True, type="primary")

# ---------- Recommendation Engine: maps a risk-increasing feature to a concrete action ----------
def recommend_for_factor(feature_name):
    """Return a targeted business action for a given SHAP top-risk feature, or None if no rule matches."""
    rules = [
        ("Contract_Month-to-month", "Offer an incentive to switch from a month-to-month plan to a 1- or 2-year contract (contract length is strongly linked to retention)."),
        ("PaymentMethod_Electronic check", "Encourage the customer to switch to automatic payment (bank transfer or credit card) instead of electronic check."),
        ("InternetService_Fiber optic", "Review the Fiber optic pricing/service experience for this customer segment — it shows a higher churn association."),
        ("MonthlyCharges", "Offer a discount or a more cost-effective plan to reduce the monthly bill."),
        ("TotalCharges", "Review the customer's overall billing history and consider a loyalty discount."),
        ("tenure", "This is a newer customer — prioritize an onboarding check-in and early engagement offers."),
        ("OnlineSecurity_No internet service", "Recommend adding Online Security as a value-added service."),
        ("TechSupport_No internet service", "Recommend adding Tech Support to improve service satisfaction."),
        ("PaperlessBilling_Yes", "Confirm the customer is comfortable with paperless billing; offer support if needed."),
    ]
    for key, action in rules:
        if key in feature_name:
            return action
    return None


if predict_clicked:
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    prob = model.predict_proba(input_encoded)[0][1] * 100

    # ---- SHAP: top factors driving this specific prediction ----
    shap_values_input = explainer.shap_values(input_encoded)
    shap_df = pd.DataFrame({
        "Feature": input_encoded.columns,
        "Impact": shap_values_input[0]
    })
    shap_df_sorted = shap_df.reindex(shap_df["Impact"].abs().sort_values(ascending=False).index)
    top_factors = shap_df_sorted.head(3)

    # Risk-increasing factors only, ranked by impact — used to drive recommendations
    risk_factors = shap_df_sorted[shap_df_sorted["Impact"] > 0].head(5)

    risk_color = "#e74c3c" if prob > 50 else "#2ecc71"
    risk_label = "High risk of churn" if prob > 50 else "Low risk of churn"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [0, 50], "color": "#e8f8f0"},
                {"range": [50, 100], "color": "#fdecea"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))

    result_col1, result_col2, result_col3 = st.columns([1, 1.5, 1])
    with result_col2:
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"<p style='text-align:center; font-size:16px; font-weight:600; color:{risk_color};'>{risk_label}</p>", unsafe_allow_html=True)
        st.progress(int(prob))

    st.markdown("**Top factors driving this prediction:**")
    for _, row in top_factors.iterrows():
        direction = "increases risk" if row["Impact"] > 0 else "decreases risk"
        st.write(f"- **{row['Feature']}** — {direction}")

    # ---- Business Recommendation Engine, driven by the actual SHAP risk factors ----
    st.markdown("**Recommended action:**")

    if prob <= 50:
        st.success("Low risk — no immediate action needed, continue regular monitoring.")
    else:
        matched_actions = []
        for _, row in risk_factors.iterrows():
            action = recommend_for_factor(row["Feature"])
            if action and action not in matched_actions:
                matched_actions.append(action)
            if len(matched_actions) == 3:
                break

        if not matched_actions:
            matched_actions.append("Contact the customer proactively for a retention check-in.")

        urgency = "Very high risk" if prob > 70 else "Moderate-to-high risk"
        urgency_icon = "🔴" if prob > 70 else "🟠"
        st.write(f"{urgency_icon} **{urgency}** — based on this customer's specific risk drivers, recommended actions:")
        for action in matched_actions:
            st.write(f"- {action}")

        #  cd notebooks
        #  streamlit run app.py
        #  venv\Scripts\activate