import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CashGuard AI",
    page_icon="💰",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.cg-card {
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    margin-bottom: 1rem;
}
.cg-title {font-size: 2.2rem; font-weight: 750; margin-bottom: 0;}
.cg-subtitle {opacity: .75; margin-top: .25rem; margin-bottom: 1.5rem;}
.cg-note {opacity: .72; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="cg-title">💰 CashGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cg-subtitle">Predict next-month cash-flow risk and turn it into practical spending guidance.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Load trained model + features
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, "cashguard_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "cashguard_features.pkl"))
    return model, features

try:
    model, FEATURES = load_assets()
except Exception as e:
    st.error(
        "Could not load the CashGuard model files. Make sure "
        "`cashguard_model.pkl` and `cashguard_features.pkl` are in the same "
        "folder as `app.py`."
    )
    st.exception(e)
    st.stop()

# -----------------------------
# Input
# -----------------------------
st.subheader("Enter your last 3 months of financial activity")
st.caption("Use the oldest month as Month 1 and the most recent month as Month 3.")

default_values = [
    (60000.0, 42000.0, 100000.0, False),
    (55000.0, 47000.0, 108000.0, False),
    (48000.0, 52000.0, 104000.0, True),
]

months = []

for i in range(3):
    st.markdown(f"**Month {i + 1}**")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    income = c1.number_input(
        "Income (₹)",
        min_value=0.0,
        value=default_values[i][0],
        step=1000.0,
        key=f"income_{i}",
    )
    expense = c2.number_input(
        "Expenses (₹)",
        min_value=0.0,
        value=default_values[i][1],
        step=1000.0,
        key=f"expense_{i}",
    )
    balance = c3.number_input(
        "Available balance (₹)",
        value=default_values[i][2],
        step=1000.0,
        key=f"balance_{i}",
        help="Prototype input representing the available balance recorded for that month.",
    )
    unexpected = c4.checkbox(
        "Unexpected expense",
        value=default_values[i][3],
        key=f"unexpected_{i}",
    )

    months.append({
        "Income": float(income),
        "Expense": float(expense),
        "Balance": float(balance),
        "Unexpected": int(unexpected),
    })

st.divider()

# -----------------------------
# Feature engineering
# Must mirror the Colab training logic.
# -----------------------------
def build_features(rows):
    incomes = np.array([x["Income"] for x in rows], dtype=float)
    expenses = np.array([x["Expense"] for x in rows], dtype=float)

    avg_income = float(incomes.mean())
    avg_expense = float(expenses.mean())

    # pandas rolling std() uses sample standard deviation (ddof=1)
    income_volatility = float(pd.Series(incomes).std(ddof=1))

    income_trend = (
        float((incomes[-1] - incomes[0]) / incomes[0])
        if incomes[0] != 0 else 0.0
    )
    expense_trend = (
        float((expenses[-1] - expenses[0]) / expenses[0])
        if expenses[0] != 0 else 0.0
    )

    # Training used previous-month balance divided by the 3-month average expense.
    cash_buffer = (
        float(rows[-1]["Balance"] / avg_expense)
        if avg_expense > 0 else 0.0
    )

    negative_cashflow_months = int(
        sum(x["Expense"] > x["Income"] for x in rows)
    )
    unexpected_expenses = int(sum(x["Unexpected"] for x in rows))

    values = {
        "Avg_Income_3M": avg_income,
        "Avg_Expense_3M": avg_expense,
        "Income_Volatility_3M": income_volatility,
        "Income_Trend_3M": income_trend,
        "Expense_Trend_3M": expense_trend,
        "Cash_Buffer": cash_buffer,
        "Negative_CashFlow_Months_3M": negative_cashflow_months,
        "Unexpected_Expenses_3M": unexpected_expenses,
    }

    return pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES), values


def money(value):
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.0f}"


def recommendation(risk, expected_income, expected_expense, reduction):
    if risk == "High":
        return (
            f"Cash-flow pressure is likely next month. Prioritize essential "
            f"expenses and review discretionary spending. Based on this prototype, "
            f"reducing planned expenses by about {money(reduction)} would create "
            f"more room for the recommended reserve."
        )
    if risk == "Medium":
        return (
            f"Your projected spending is close to or above your expected income. "
            f"Consider reviewing non-essential expenses and preserving additional "
            f"cash before making discretionary purchases."
        )
    return (
        "Your projected income currently covers expected expenses with more room "
        "than the higher-risk profiles. Continue monitoring income changes, "
        "unexpected expenses and monthly cash flow."
    )


if st.button("Analyze My Cash Flow", type="primary", use_container_width=True):
    X, engineered = build_features(months)

    prediction = str(model.predict(X)[0])

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, probs)
        }

    # Same simple trend-adjusted estimation used in the Colab prototype.
    expected_income = max(
        0.0,
        engineered["Avg_Income_3M"]
        * (1 + engineered["Income_Trend_3M"] / 3)
    )
    expected_expense = max(
        0.0,
        engineered["Avg_Expense_3M"]
        * (1 + engineered["Expense_Trend_3M"] / 3)
    )

    expected_cash_flow = expected_income - expected_expense
    coverage = (
        expected_income / expected_expense
        if expected_expense > 0 else np.nan
    )

    # Transparent prototype decision rules, not validated financial advice.
    reserve_rates = {"Low": 0.10, "Medium": 0.15, "High": 0.20}
    reserve_rate = reserve_rates.get(prediction, 0.10)
    recommended_reserve = max(0.0, expected_income * reserve_rate)

    safe_to_spend = max(
        0.0,
        expected_income - expected_expense - recommended_reserve
    )
    affordable_expense = max(0.0, expected_income - recommended_reserve)
    expense_reduction = max(0.0, expected_expense - affordable_expense)

    st.divider()
    st.subheader("Your CashGuard analysis")

    risk_display = {
        "High": "🔴 HIGH",
        "Medium": "🟠 MEDIUM",
        "Low": "🟢 LOW",
    }.get(prediction, prediction.upper())

    a, b, c = st.columns(3)
    a.metric("Next-month risk", risk_display)

    if probabilities:
        a.caption(
            "Model confidence: "
            + " · ".join(
                f"{k} {v:.0%}"
                for k, v in sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            )
        )

    b.metric("Expected income", money(expected_income))
    c.metric("Expected expenses", money(expected_expense))

    a, b, c = st.columns(3)
    a.metric("Expected cash flow", money(expected_cash_flow))
    b.metric(
        "Cash-flow coverage",
        f"{coverage:.2f}x" if np.isfinite(coverage) else "N/A"
    )
    c.metric("Safe to spend", money(safe_to_spend))

    st.subheader("Your CashGuard plan")
    a, b = st.columns(2)
    a.metric("Recommended reserve", money(recommended_reserve))
    b.metric("Expense reduction target", money(expense_reduction))

    st.markdown("#### Why did the model flag this profile?")
    reasons = []

    if engineered["Negative_CashFlow_Months_3M"] > 0:
        reasons.append(
            f"Expenses exceeded income in "
            f"{engineered['Negative_CashFlow_Months_3M']:.0f} of the last 3 months."
        )
    if engineered["Income_Trend_3M"] < 0:
        reasons.append(
            f"Income declined by "
            f"{abs(engineered['Income_Trend_3M']) * 100:.1f}% from Month 1 to Month 3."
        )
    if engineered["Expense_Trend_3M"] > 0:
        reasons.append(
            f"Expenses increased by "
            f"{engineered['Expense_Trend_3M'] * 100:.1f}% from Month 1 to Month 3."
        )
    if engineered["Unexpected_Expenses_3M"] > 0:
        reasons.append(
            f"{engineered['Unexpected_Expenses_3M']:.0f} unexpected-expense "
            f"event(s) were reported."
        )
    if engineered["Income_Volatility_3M"] > 0.15 * max(
        engineered["Avg_Income_3M"], 1
    ):
        reasons.append("Income varied substantially across the three-month period.")

    if not reasons:
        reasons.append(
            "The model assessed the combined pattern of recent income, expenses, "
            "volatility and cash-flow behavior."
        )

    for reason in reasons:
        st.write(f"• {reason}")

    st.markdown("#### Suggested next action")
    st.info(
        recommendation(
            prediction,
            expected_income,
            expected_expense,
            expense_reduction,
        )
    )

    with st.expander("View model inputs"):
        display_features = pd.DataFrame({
            "Feature": FEATURES,
            "Value": [engineered[f] for f in FEATURES],
        })
        st.dataframe(display_features, use_container_width=True, hide_index=True)

    st.caption(
        "Prototype only. CashGuard AI uses a model trained on synthetic data and "
        "transparent decision-support assumptions. Outputs are for demonstration "
        "and are not financial advice."
    )
