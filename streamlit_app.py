import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Field Drop Retention Console (Demo)", layout="wide")

# ----------------------------
# Synthetic data generator
# ----------------------------
def generate_demo_data(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_groups = ["Less_Than_20", "20_To_35", "35_To_50", "50_To_65", "Greater_Than_65", "Unknown"]
    personas = ["301 Savvy Shoppers", "508 Frugal Living", "Other"]
    arrears = ["None", "Catch_All", "Medium", "High"]
    segments = ["Exception", "Mass"]

    df = pd.DataFrame({
        "supporter_id": [f"S{100000+i}" for i in range(n)],
        "segment": rng.choice(segments, size=n, p=[0.65, 0.35]),
        "age_group": rng.choice(age_groups, size=n, p=[0.03, 0.20, 0.30, 0.22, 0.15, 0.10]),
        "helix_persona": rng.choice(personas, size=n, p=[0.22, 0.18, 0.60]),
        "dnar_flag": rng.choice([0, 1], size=n, p=[0.78, 0.22]),
        "arrears_status": rng.choice(arrears, size=n, p=[0.62, 0.14, 0.14, 0.10]),
        "has_sms": rng.choice([0, 1], size=n, p=[0.40, 0.60]),
        "valid_channels": rng.integers(1, 4, size=n),  # 1 to 3: Email, Phone, Direct Mail
        "sponsorship_years": np.clip(rng.normal(7.5, 4.0, size=n), 0.3, 18.0).round(1),
        "avg_active_child_age": np.clip(rng.normal(14.0, 5.0, size=n), 1.0, 26.0).round(1),
        "letters_to_dropped_2y": np.clip(rng.poisson(2.0, size=n), 0, 12),
        "letters_to_others_2y": np.clip(rng.poisson(3.0, size=n), 0, 20),
        "num_donations": np.clip(rng.normal(85, 45, size=n), 0, 240).round(0).astype(int),
        "avg_donation_amount": np.clip(rng.normal(18, 9, size=n), 5, 120).round(0).astype(int),
        "child_transitions": np.clip(rng.poisson(1.2, size=n), 0, 6),
    })

    # Impose a few realistic relationships to make the demo coherent
    # 508 Frugal Living skew older
    mask_frugal = df["helix_persona"] == "508 Frugal Living"
    df.loc[mask_frugal, "age_group"] = rng.choice(["50_To_65", "Greater_Than_65"], size=mask_frugal.sum(), p=[0.35, 0.65])

    # 301 Savvy Shoppers skew mid life
    mask_savvy = df["helix_persona"] == "301 Savvy Shoppers"
    df.loc[mask_savvy, "age_group"] = rng.choice(["20_To_35", "35_To_50", "50_To_65"], size=mask_savvy.sum(), p=[0.20, 0.60, 0.20])

    # Valid channels align with has_sms a bit
    df.loc[df["has_sms"] == 1, "valid_channels"] = rng.choice([2, 3], size=(df["has_sms"] == 1).sum(), p=[0.35, 0.65])
    df.loc[df["has_sms"] == 0, "valid_channels"] = rng.choice([1, 2], size=(df["has_sms"] == 0).sum(), p=[0.65, 0.35])

    # Risk score: build a synthetic "model score" driven by your key insights
    risk = np.zeros(n, dtype=float)

    # Exception flags
    risk += df["dnar_flag"] * 0.18
    risk += df["arrears_status"].map({"None": 0.00, "Catch_All": 0.06, "Medium": 0.10, "High": 0.16}).to_numpy()

    # Engagement pattern: letters to dropped vs other children
    risk += np.clip(df["letters_to_dropped_2y"] / 10.0, 0, 0.18)
    risk -= np.clip(df["letters_to_others_2y"] / 25.0, 0, 0.16)

    # Long bond increases risk
    risk += np.clip((df["sponsorship_years"] - 6.0) / 30.0, 0, 0.20)

    # Portfolio age increases risk
    risk += np.clip((df["avg_active_child_age"] - 15.0) / 25.0, 0, 0.18)

    # Donation pattern: 90 to 160 donations risk band, >160 protective, <50 with higher avg protective
    risk += np.where((df["num_donations"] >= 90) & (df["num_donations"] <= 160), 0.10, 0.0)
    risk -= np.where(df["num_donations"] > 160, 0.10, 0.0)
    risk -= np.where((df["num_donations"] < 50) & (df["avg_donation_amount"] > 10), 0.05, 0.0)

    # Age group effect: 35 to 50 protective, 65+ higher risk, 20 to 35 mixed (keep near neutral)
    risk += df["age_group"].map({
        "Less_Than_20": 0.04,
        "20_To_35": 0.00,
        "35_To_50": -0.06,
        "50_To_65": 0.03,
        "Greater_Than_65": 0.09,
        "Unknown": 0.01
    }).to_numpy()

    # Persona aligns with age story
    risk += df["helix_persona"].map({
        "301 Savvy Shoppers": -0.03,
        "508 Frugal Living": 0.04,
        "Other": 0.00
    }).to_numpy()

    # Contact channel coverage as a risk signal
    risk += np.where(df["valid_channels"] == 3, 0.03, 0.0)

    # Add a bit of noise
    risk += rng.normal(0, 0.02, size=n)

    # Squash to 0 to 1
    df["churn_risk"] = np.clip(1 / (1 + np.exp(-5 * (risk - 0.15))), 0, 1).round(3)

    return df


def build_why_flagged(row: pd.Series) -> list[str]:
    reasons = []

    # Engagement pattern
    if row["letters_to_dropped_2y"] >= 4 and row["letters_to_others_2y"] == 0:
        reasons.append("Engagement concentrated in dropped child (letters to dropped, none to other children)")
    elif row["letters_to_others_2y"] >= 8:
        reasons.append("Broad engagement across other children (protective signal)")

    # Long bond
    if row["sponsorship_years"] >= 10:
        reasons.append("Long sponsorship bond (10+ years)")

    # Portfolio age
    if row["avg_active_child_age"] >= 15:
        reasons.append("Older active child portfolio (average age 15+)")

    # Exception flags
    if row["dnar_flag"] == 1:
        reasons.append("DNAR flag present (Do Not Auto Replace)")

    if row["arrears_status"] in ["Medium", "High"]:
        reasons.append(f"Arrears status at drop: {row['arrears_status']} (active missed payment risk)")
    elif row["arrears_status"] == "Catch_All":
        reasons.append("Behind on payments (catch all arrears)")

    # Donation pattern
    if 90 <= row["num_donations"] <= 160:
        reasons.append("High donation frequency band (90 to 160) linked to higher churn sensitivity")
    if row["num_donations"] > 160:
        reasons.append("Very high donation frequency (160+) linked to stronger retention")

    # Persona and age
    if row["age_group"] == "Greater_Than_65":
        reasons.append("Age group 65+ (higher churn segment in Field Drop)")
    if row["helix_persona"] == "508 Frugal Living":
        reasons.append("Helix persona: 508 Frugal Living (higher churn segment)")
    if row["helix_persona"] == "301 Savvy Shoppers":
        reasons.append("Helix persona: 301 Savvy Shoppers (more resilient segment)")

    # Contactability
    if row["valid_channels"] == 3:
        reasons.append("High channel coverage (email, phone, direct mail all valid)")

    # Keep top 3 to 5
    if not reasons:
        reasons = ["Multiple small signals combined (no single dominant driver)"]
    return reasons[:5]


def recommended_call_focus(row: pd.Series) -> list[str]:
    actions = []

    # High empathy cases
    if row["sponsorship_years"] >= 10 or row["letters_to_dropped_2y"] >= 4:
        actions.append("Lead with empathy and acknowledge the loss or change")

    # Financial support
    if row["arrears_status"] in ["Medium", "High"]:
        actions.append("Address payment concerns early and offer support options")
    elif row["arrears_status"] == "Catch_All":
        actions.append("Confirm payment expectations and offer an easy path to get back on track")

    # Replacement framing
    actions.append("Frame replacement as continuity of impact, not a swap")

    # Engagement style
    if row["letters_to_others_2y"] == 0:
        actions.append("Use a child centred approach, focus on meaning and reassurance")
    else:
        actions.append("Use mission framing, reinforce their broader program connection")

    # Channel preference
    actions.append("Confirm preferred channel and next step (phone, email, direct mail)")

    return actions[:5]


# ----------------------------
# App UI
# ----------------------------
st.title("Field Drop Retention Console (Demo)")
st.caption("Demo data only. This is a prototype to showcase how consultants could prioritise high risk supporters and understand why they are flagged.")

with st.sidebar:
    st.header("Filters")
    n = st.slider("Number of demo supporters", 100, 800, 300, 50)
    seed = st.number_input("Random seed", value=7, step=1)

    min_risk = st.slider("Minimum churn risk", 0.0, 1.0, 0.60, 0.05)
    seg = st.selectbox("Segment", ["All", "Exception", "Mass"])
    dnar = st.selectbox("DNAR", ["All", "DNAR only", "Non DNAR only"])
    arrears = st.selectbox("Arrears status", ["All", "High", "Medium", "Catch_All", "None"])
    persona = st.selectbox("Helix persona", ["All", "301 Savvy Shoppers", "508 Frugal Living", "Other"])
    age = st.selectbox("Age group", ["All", "Less_Than_20", "20_To_35", "35_To_50", "50_To_65", "Greater_Than_65", "Unknown"])

df = generate_demo_data(n=int(n), seed=int(seed))

# Apply filters
view = df.copy()
view = view[view["churn_risk"] >= min_risk]

if seg != "All":
    view = view[view["segment"] == seg]

if dnar == "DNAR only":
    view = view[view["dnar_flag"] == 1]
elif dnar == "Non DNAR only":
    view = view[view["dnar_flag"] == 0]

if arrears != "All":
    view = view[view["arrears_status"] == arrears]

if persona != "All":
    view = view[view["helix_persona"] == persona]

if age != "All":
    view = view[view["age_group"] == age]

view = view.sort_values("churn_risk", ascending=False)

# Queue summary
top_left, top_mid, top_right = st.columns(3)
with top_left:
    st.metric("Supporters in view", f"{len(view):,}")
with top_mid:
    st.metric("Average churn risk", f"{view['churn_risk'].mean():.2f}" if len(view) else "n/a")
with top_right:
    st.metric("High risk (>= 0.80)", f"{(view['churn_risk'] >= 0.80).sum():,}" if len(view) else "0")

st.subheader("Exception queue (sorted by churn risk)")

display_cols = [
    "supporter_id", "segment", "churn_risk", "dnar_flag", "arrears_status",
    "age_group", "helix_persona", "sponsorship_years", "avg_active_child_age",
    "letters_to_dropped_2y", "letters_to_others_2y", "num_donations", "avg_donation_amount",
    "valid_channels"
]

st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

st.divider()

if len(view) == 0:
    st.info("No supporters match the current filters. Try lowering the minimum churn risk.")
    st.stop()

selected_id = st.selectbox("Select a supporter to review", view["supporter_id"].tolist(), index=0)
row = view.loc[view["supporter_id"] == selected_id].iloc[0]

left, right = st.columns([1, 1])

with left:
    st.subheader("Supporter profile")
    st.write(
        {
            "Supporter ID": row["supporter_id"],
            "Segment": row["segment"],
            "Churn risk score": float(row["churn_risk"]),
            "DNAR": "Yes" if int(row["dnar_flag"]) == 1 else "No",
            "Arrears status": row["arrears_status"],
            "Age group": row["age_group"],
            "Helix persona": row["helix_persona"],
            "Sponsorship years": float(row["sponsorship_years"]),
            "Average active child age": float(row["avg_active_child_age"]),
            "Letters to dropped child (2Y)": int(row["letters_to_dropped_2y"]),
            "Letters to other children (2Y)": int(row["letters_to_others_2y"]),
            "Number of donations": int(row["num_donations"]),
            "Average donation amount": int(row["avg_donation_amount"]),
            "Valid contact channels (email, phone, direct mail)": int(row["valid_channels"]),
        }
    )

with right:
    st.subheader("Why flagged (top drivers)")
    for r in build_why_flagged(row):
        st.write("• " + r)

    st.subheader("Suggested call focus")
    for a in recommended_call_focus(row):
        st.write("• " + a)

    st.subheader("Outcome (demo)")
    outcome = st.selectbox("Log outcome", ["Not logged", "Contacted, continued", "Contacted, cancelled", "No contact, follow up email", "No contact, follow up direct mail"])
    st.caption("This is for demo only. In a real tool, this would feed monitoring and retraining.")

