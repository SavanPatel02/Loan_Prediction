import streamlit as st
import requests

# PAGE CONFIG

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)


# CUSTOM STYLE

st.markdown("""
<style>
.big-title {
    font-size:32px;
    font-weight:700;
    color:#1f77b4;
}
.card {
    padding:25px;
    border-radius:12px;
    background-color:#f5f7fa;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}
.metric-box {
    font-size:20px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown('<p class="big-title">💰 Loan Default Prediction System</p>', unsafe_allow_html=True)
st.write("Fill applicant details below to predict default risk.")

st.divider()


# FORM LAYOUT

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")

    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Income = st.number_input("Income", min_value=0)
    Education = st.selectbox("Education", ["High School", "Bachelor", "Master"])
    EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])

with col2:
    st.subheader("🏦 Loan Details")

    LoanAmount = st.number_input("Loan Amount", min_value=0)
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    MonthsEmployed = st.number_input("Months Employed", min_value=0)
    NumCreditLines = st.number_input("Number of Credit Lines", min_value=0)
    InterestRate = st.number_input("Interest Rate", min_value=0.0)
    LoanTerm = st.number_input("Loan Term", min_value=0)
    DTIRatio = st.number_input("DTI Ratio", min_value=0.0)

HasMortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
HasDependents = st.selectbox("Has Dependents", ["Yes", "No"])
LoanPurpose = st.selectbox("Loan Purpose", ["Home", "Car", "Business"])
HasCoSigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

st.divider()


# PREDICTION BUTTON

if st.button("🚀 Predict Default Risk", use_container_width=True):

    payload = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "Education": Education,
        "EmploymentType": EmploymentType,
        "MaritalStatus": MaritalStatus,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "LoanPurpose": LoanPurpose,
        "HasCoSigner": HasCoSigner
    }

    try:
        with st.spinner("🔎 Predicting..."):
            r = requests.post("http://api:8000/predict", json=payload, timeout=20)

        if r.status_code == 200:

            result = r.json()

            st.success("Prediction Completed Successfully!")

           
            # PROBABILITY CONVERSION
            
            prob_default = result["default_probability"]
            rejection_percent = prob_default * 100
            approval_percent = (1 - prob_default) * 100

            
            # RISK CATEGORY
            
            if prob_default < 0.30:
                risk_level = "Low Risk"
                risk_color = "green"
            elif prob_default < 0.60:
                risk_level = "Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "High Risk"
                risk_color = "red"

           
            # DISPLAY RESULT
          
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("📊 Loan Approval Analysis")

            colA, colB = st.columns(2)

            with colA:
                st.metric("✅ Approval Chance", f"{approval_percent:.2f}%")

            with colB:
                st.metric("❌ Rejection Chance", f"{rejection_percent:.2f}%")

            st.markdown(f"### 🎯 Risk Category: **{risk_level}**")

           
            # REASONING LOGIC
            
            st.subheader("📌 Key Decision Factors")

            reasons = []

            if CreditScore < 600:
                reasons.append("Low credit score reduces approval likelihood.")

            if Income < 30000:
                reasons.append("Low income compared to loan amount.")

            if LoanAmount > Income * 5:
                reasons.append("Loan amount is high relative to income.")

            if DTIRatio > 0.4:
                reasons.append("High Debt-to-Income ratio.")

            if MonthsEmployed < 12:
                reasons.append("Short employment history.")

            if len(reasons) == 0:
                reasons.append("Strong financial and credit profile.")

            for reason in reasons:
                st.write("•", reason)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error(f"API Error: {r.json()}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection failed: {str(e)}")
