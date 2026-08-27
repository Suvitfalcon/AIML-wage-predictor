import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("labour_dataset.csv")

# Keep original dataframe for dropdowns

df_original = df.copy()

# -----------------------------
# LABEL ENCODING
# -----------------------------

le_city = LabelEncoder()
le_state = LabelEncoder()
le_skill = LabelEncoder()

df['city'] = le_city.fit_transform(df['city'])
df['state'] = le_state.fit_transform(df['state'])
df['skill'] = le_skill.fit_transform(df['skill'])

# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = df.drop('wage', axis=1)
y = df['wage']

# -----------------------------
# SCALE FEATURES
# -----------------------------

numeric_cols = [
    'experience',
    'demand_index',
    'cost_of_living'
]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ---
# TRAIN MODEL
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="Labour Wage Predictor",
    layout="centered"
)

st.title("Labour Wage Prediction System")

st.write("Enter worker details")

# -----------------------------
# STATE DROPDOWN
# -----------------------------

states = sorted(df_original['state'].unique())

state = st.selectbox(
    "Select State",
    states
)

# -----------------------------
# CITY DROPDOWN (depends on state)
# -----------------------------

cities = sorted(
    df_original[df_original['state'] == state]['city'].unique()
)

# hew sretbsk

city = st.selectbox(
    "Select City",
    cities
)

# -----------------------------
# kjbcckdc
# SKILL DROPDOWN (depends on state + city)
# -----------------------------

skills = sorted(
    df_original[
        (df_original['state'] == state) &
        (df_original['city'] == city)
    ]['skill'].unique()
)

skill = st.selectbox(
    "Select Skill",
    skills
)

# --------------------------
# OTHER INPUTS
# -------------------
# kjnad

experience = st.slider(
    "Experience (Years)",
    0,
    30,
    5
)

demand_index = st.slider(
    "Demand Index",
    0,
    100,
    50
)

cost_of_living = st.slider(
    "Cost of Living",
    0,
    100,
    50
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------

if st.button("Predict Wage"):

    city_encoded = le_city.transform([city])[0]
    state_encoded = le_state.transform([state])[0]
    skill_encoded = le_skill.transform([skill])[0]

    input_data = pd.DataFrame({
        'city': [city_encoded],
        'state': [state_encoded],
        'skill': [skill_encoded],
        'experience': [experience],
        'demand_index': [demand_index],
        'cost_of_living': [cost_of_living]
    })

    input_data[numeric_cols] = scaler.transform(
        input_data[numeric_cols]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Wage: ₹{prediction[0]:,.2f}"
    )