
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Energy Intelligence",
    page_icon="⚡",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
        color: #1f2937;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        background-color: #e8f3ed;
        border-right: 1px solid #d1e5d8;
    }

    [data-testid="stSidebar"] * {
        color: #163b2a !important;
    }

    .stApp p,
    .stApp label,
    .stApp li {
        color: #263238;
    }

    h1, h2, h3 {
        color: #14532d !important;
        font-weight: 700 !important;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #166534;
    }

    .subtitle {
        font-size: 16px;
        color: #52665a;
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7df;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(20, 83, 45, 0.05);
    }

    div[data-testid="stMetric"] label {
        color: #52665a !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #14532d !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="select"] * {
        color: #1f2937 !important;
    }

    .energy-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #ffffff;
        color: #1f2937;
        font-size: 14px;
    }

    .energy-table th {
        background-color: #dcefe3;
        color: #14532d;
        padding: 12px;
        text-align: left;
    }

    .energy-table td {
        color: #263238;
        padding: 10px;
        border-bottom: 1px solid #e2e8f0;
    }

    .energy-table tr:nth-child(even) {
        background-color: #f8fafc;
    }

    .stButton button {
        background-color: #166534;
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        padding: 8px 18px;
    }

    .stButton button:hover {
        background-color: #14532d;
        color: #ffffff !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER: HTML TABLE
# =====================================================

def show_table(data):
    html = data.to_html(
        index=False,
        border=0,
        classes="energy-table",
        escape=True
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# =====================================================
# HELPER: LOAD DATA
# =====================================================

DATA_PATH = Path(__file__).parent / "sample_energy.csv"


@st.cache_data
def load_data(file_path, uploaded_file=None):

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(file_path)

    required_columns = {
        "timestamp",
        "consumption"
    }

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain 'timestamp' and "
            "'consumption' columns."
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["consumption"] = pd.to_numeric(
        data["consumption"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "consumption"]
    )

    data = data[
        data["consumption"] >= 0
    ].copy()

    data = data.sort_values("timestamp")

    return data


# =====================================================
# HELPER: DAILY DATA
# =====================================================

def prepare_daily_data(data):

    daily_data = (
        data.set_index("timestamp")["consumption"]
        .resample("D")
        .sum()
        .dropna()
    )

    return daily_data


# =====================================================
# HELPER: NEXT-DAY FORECAST
# NumPy least-squares regression
# =====================================================

def forecast_next_day(daily):

    values = daily.to_numpy(dtype=float)

    if len(values) < 10:
        return None, None

    X = []
    y = []

    # Previous day's total + previous 7-day average
    for i in range(7, len(values)):

        X.append([
            values[i - 1],
            values[i - 7:i].mean()
        ])

        y.append(values[i])

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(X) < 3:
        return None, None

    # Chronological train/test split
    split = int(len(X) * 0.8)
    split = max(1, min(split, len(X) - 1))

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    X_train_b = np.column_stack([
        np.ones(len(X_train)),
        X_train
    ])

    coefficients = np.linalg.lstsq(
        X_train_b,
        y_train,
        rcond=None
    )[0]

    X_test_b = np.column_stack([
        np.ones(len(X_test)),
        X_test
    ])

    test_predictions = X_test_b @ coefficients

    mae = float(
        np.mean(
            np.abs(y_test - test_predictions)
        )
    )

    latest_features = np.array([
        1.0,
        values[-1],
        values[-7:].mean()
    ])

    next_prediction = float(
        latest_features @ coefficients
    )

    return max(0, next_prediction), mae


# =====================================================
# PREDICTIVE MODE FORECAST
# =====================================================

def forecast_energy(daily, forecast_days):

    daily = daily.dropna().astype(float)

    if len(daily) < 14:
        return None, None

    # Use up to the latest 60 days
    recent = daily.tail(
        min(60, len(daily))
    )

    y = recent.to_numpy()
    x = np.arange(
        len(y),
        dtype=float
    )

    # Estimate linear trend
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    # Average of the latest 7 days
    weekly_average = float(
        daily.tail(7).mean()
    )

    future_x = np.arange(
        len(y),
        len(y) + forecast_days,
        dtype=float
    )

    trend_forecast = (
        intercept + slope * future_x
    )

    # Blend trend with recent average
    predictions = (
        0.5 * trend_forecast
        + 0.5 * weekly_average
    )

    predictions = np.maximum(
        predictions,
        0
    )

    future_dates = pd.date_range(
        start=daily.index[-1]
        + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast = pd.Series(
        predictions,
        index=future_dates,
        name="Predicted consumption"
    )

    # Simple historical-average baseline validation.
    # This MAE is NOT the trend forecast's MAE.
    split = int(len(daily) * 0.8)

    train = daily.iloc[:split]
    test = daily.iloc[split:]

    if len(train) > 0 and len(test) > 0:

        baseline_prediction = float(
            train.tail(7).mean()
        )

        mae = float(
            np.mean(
                np.abs(
                    test.to_numpy()
                    - baseline_prediction
                )
            )
        )

    else:
        mae = None

    return forecast, mae


# =====================================================
# PAGE: PREDICTIVE MODE
# =====================================================

def show_predictive_mode(df):

    st.header("🔮 Predictive Mode")

    st.write(
        "Estimate future energy consumption using "
        "historical daily usage patterns."
    )

    daily = prepare_daily_data(df)

    if len(daily) < 14:

        st.warning(
            "Please provide at least 14 days "
            "of valid consumption data."
        )

        return

    forecast_days = st.select_slider(
        "Forecast duration",
        options=[1, 3, 7, 14],
        value=7
    )

    forecast, mae = forecast_energy(
        daily,
        forecast_days
    )

    if forecast is None:
        st.warning("Unable to generate forecast.")
        return

    total_forecast = float(
        forecast.sum()
    )

    average_forecast = float(
        forecast.mean()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast duration",
        f"{forecast_days} days"
    )

    col2.metric(
        "Total predicted consumption",
        f"{total_forecast:,.2f}"
    )

    col3.metric(
        "Daily predicted average",
        f"{average_forecast:,.2f}"
    )

    st.caption(
        "Values are in the dataset's original units. "
        "Verify the units before interpreting them as kWh."
    )

    # Historical + forecast chart
    history = daily.tail(30)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines+markers",
            name="Historical consumption"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Predicted consumption",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Historical vs Predicted Consumption",
        xaxis_title="Date",
        yaxis_title="Consumption (dataset units)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Forecast table
    st.subheader("Forecast Details")

    forecast_table = pd.DataFrame({
        "Date": forecast.index.strftime(
            "%Y-%m-%d"
        ),
        "Predicted consumption": forecast.values
    })

    show_table(forecast_table)

    with st.expander("How does this prediction work?"):

        st.write("""
        1. Historical readings are grouped into daily totals.
        2. A linear trend is estimated from recent data.
        3. The trend is blended with the recent 7-day average.
        4. Estimates are generated for the selected future period.

        This is a simple baseline, not a trained deep-learning model.
        It does not currently use occupancy, holidays, weather,
        or equipment-level consumption.
        """)

        if mae is not None:
            st.write(
                f"Simple historical-average baseline MAE: "
                f"{mae:,.2f} dataset units per day."
            )

            st.caption(
                "This MAE evaluates a simple average baseline, "
                "not the multi-day trend forecast shown above."
            )

    st.caption(
        "Forecasts are estimates, not guaranteed future "
        "consumption. The dataset's units have not been verified."
    )


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">⚡ Energy Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Occupancy-Aware Energy Intelligence & Optimization System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚡ Energy AI")
st.sidebar.caption("Building Energy Monitoring")



page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Energy Analytics",
        "Predictive Mode",
        "AI Insights",
        "Optimization Simulator",
        "Energy Action Planner",
        "Occupancy-Aware Energy Control",
        "Wi-Fi Intelligence"
    ]
)
st.sidebar.divider()
st.sidebar.subheader("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload energy CSV",
    type=["csv"]
)


# =====================================================
# LOAD DATA
# =====================================================

try:

    if uploaded_file is not None:

        df = load_data(
            DATA_PATH,
            uploaded_file
        )

    else:

        if not DATA_PATH.exists():

            st.error(
                "sample_energy.csv not found. "
                "Place it beside app.py or upload a CSV."
            )

            st.stop()

        df = load_data(DATA_PATH)

except Exception as error:

    st.error(
        f"Could not load data: {error}"
    )

    st.stop()


if df.empty:

    st.error(
        "No valid energy readings found."
    )

    st.stop()


# =====================================================
# DATA PREPARATION
# =====================================================

df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["day_name"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month

daily = prepare_daily_data(df)

hourly = (
    df.groupby("hour")["consumption"]
    .mean()
    .reset_index()
)


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "Overview":

    st.header("📊 Energy Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Recorded Consumption",
        f"{df['consumption'].sum():,.2f}"
    )

    col2.metric(
        "Average Reading",
        f"{df['consumption'].mean():,.3f}"
    )

    col3.metric(
        "Maximum Reading",
        f"{df['consumption'].max():,.3f}"
    )

    col4.metric(
        "Total Readings",
        f"{len(df):,}"
    )

    st.caption(
        "Values are shown in the dataset's original units. "
        "Verify the unit before interpreting as kWh."
    )

    st.divider()

    st.subheader("Daily Consumption Trend")

    daily_df = daily.rename(
        "consumption"
    ).reset_index()

    fig = px.area(
        daily_df,
        x="timestamp",
        y="consumption",
        title="Daily Recorded Consumption",
        template="plotly_white"
    )

    fig.update_traces(
        line_color="#166534",
        fillcolor="rgba(22,101,52,0.18)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Hourly Average")

        fig_hour = px.bar(
            hourly,
            x="hour",
            y="consumption",
            template="plotly_white",
            labels={
                "hour": "Hour of Day",
                "consumption": "Average Reading"
            }
        )

        st.plotly_chart(
            fig_hour,
            use_container_width=True
        )

    with right:

        st.subheader("Recent Readings")

        recent = df[
            ["timestamp", "consumption"]
        ].tail(15)

        show_table(recent)


# =====================================================
# PAGE 2: ENERGY ANALYTICS
# =====================================================

elif page == "Energy Analytics":

    st.header("📈 Energy Analytics")

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()

    if min_date == max_date:

        start_date = min_date
        end_date = max_date

    else:

        selected_dates = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(selected_dates) != 2:

            st.warning(
                "Select both a start and end date."
            )

            st.stop()

        start_date, end_date = selected_dates

    filtered = df[
        (df["timestamp"].dt.date >= start_date)
        & (df["timestamp"].dt.date <= end_date)
    ].copy()

    if filtered.empty:

        st.warning(
            "No readings found in this date range."
        )

        st.stop()

    filtered_daily = prepare_daily_data(
        filtered
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Selected Total",
        f"{filtered['consumption'].sum():,.2f}"
    )

    c2.metric(
        "Average Reading",
        f"{filtered['consumption'].mean():,.3f}"
    )

    c3.metric(
        "Readings",
        f"{len(filtered):,}"
    )

    st.subheader("Consumption by Date")

    filtered_daily_df = (
        filtered_daily.rename("consumption")
        .reset_index()
    )

    fig = px.line(
        filtered_daily_df,
        x="timestamp",
        y="consumption",
        markers=True,
        template="plotly_white",
        labels={
            "timestamp": "Date",
            "consumption": "Daily Consumption"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Hourly Consumption Pattern")

    hourly_filtered = (
        filtered.groupby("hour")["consumption"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        hourly_filtered,
        x="hour",
        y="consumption",
        template="plotly_white",
        labels={
            "hour": "Hour",
            "consumption": "Average Reading"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Highest Individual Readings")

    highest = filtered.nlargest(
        10,
        "consumption"
    )[["timestamp", "consumption"]]

    show_table(highest)


# =====================================================
# PAGE 3: PREDICTIVE MODE
# =====================================================

elif page == "Predictive Mode":

    show_predictive_mode(df)


# =====================================================
# PAGE 4: AI INSIGHTS
# =====================================================

elif page == "AI Insights":

    st.header("🤖 AI Energy Forecasting")

    st.write(
        "Estimate next-day consumption from historical "
        "daily readings and flag unusual high-consumption days."
    )

    st.info(
        "The forecast is a fitted linear regression baseline "
        "implemented with NumPy. It does not require scikit-learn."
    )

    if len(daily) < 10:

        st.warning(
            "At least 10 daily observations are needed."
        )

    else:

        prediction, mae = forecast_next_day(
            daily
        )

        if prediction is not None:

            st.subheader("📈 Next-Day Prediction")

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted Next-Day Consumption",
                f"{prediction:,.3f}"
            )

            c2.metric(
                "Holdout MAE",
                f"{mae:,.3f}"
            )

            st.caption(
                "MAE is calculated on later observations "
                "held out from model fitting. It is measured "
                "in the dataset's original units."
            )

            recent = daily.tail(30).rename(
                "Actual Daily Total"
            ).reset_index()

            fig = px.line(
                recent,
                x="timestamp",
                y="Actual Daily Total",
                title="Recent Daily Consumption",
                template="plotly_white"
            )

            fig.add_hline(
                y=prediction,
                line_dash="dash",
                annotation_text="Next-Day Prediction"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    st.subheader("🚨 Unusual Consumption Detection")

    rolling_mean = daily.rolling(
        7,
        min_periods=3
    ).mean()

    rolling_std = daily.rolling(
        7,
        min_periods=3
    ).std()

    upper_threshold = (
        rolling_mean + 2 * rolling_std
    )

    anomalies = daily[
        (daily > upper_threshold)
        & upper_threshold.notna()
    ]

    if anomalies.empty:

        st.success(
            "No unusually high daily totals were flagged "
            "by this threshold."
        )

    else:

        st.warning(
            f"{len(anomalies)} high-consumption days "
            "were flagged for investigation."
        )

        anomaly_table = anomalies.rename(
            "Consumption"
        ).reset_index()

        show_table(anomaly_table)

        
        # Simple investigation of a flagged date
        st.subheader("🔍 Investigate a Flagged Day")

        selected_date = st.selectbox(
            "Choose a flagged date",
            anomalies.index.strftime("%Y-%m-%d")
        )

        selected_date = pd.Timestamp(selected_date)

        selected_value = daily.loc[selected_date]

        previous_days = daily[
            (daily.index < selected_date)
        ].tail(7)

        if len(previous_days) > 0:
            previous_average = previous_days.mean()

            difference = selected_value - previous_average

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Selected Day",
                f"{selected_value:,.3f}"
            )

            c2.metric(
                "Previous 7 Available Days Average",
                f"{previous_average:,.3f}"
            )

            c3.metric(
                "Difference from Average",
                f"{difference:+,.3f}"
            )

            comparison = pd.DataFrame({
                "Period": [
                    "Previous 7 available days",
                    "Selected flagged day"
                ],
                "Consumption": [
                    previous_average,
                    selected_value
                ]
            })

            fig_compare = px.bar(
                comparison,
                x="Period",
                y="Consumption",
                title="Flagged Day vs Previous Days",
                template="plotly_white"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            st.info(
                "This comparison shows how the selected day "
                "differs from recent recorded usage. It does "
                "not identify the cause of the increase."
            )

        else:
            st.warning(
                "Not enough earlier data to compare this date."
            )

        fig = px.scatter(
            anomaly_table,
            x="timestamp",
            y="Consumption",
            title="Flagged High-Consumption Days",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "An anomaly is an unusual reading, not proof "
        "of energy waste or an equipment fault."
    )


# =====================================================
# PAGE 5: OPTIMIZATION SIMULATOR
# =====================================================

elif page == "Optimization Simulator":

    st.header("🌱 Energy Optimization Simulator")

    st.write(
        "Explore hypothetical energy reductions using "
        "the recorded data and your own assumptions."
    )

    st.warning(
        "Simulation only: this app does not control "
        "real building equipment."
    )

    total_consumption = float(
        df["consumption"].sum()
    )

    st.subheader("Scenario Settings")

    col1, col2 = st.columns(2)

    with col1:

        reduction = st.slider(
            "Hypothetical Reduction (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        )

    with col2:

        tariff = st.number_input(
            "Assumed Tariff (₹ per unit)",
            min_value=0.0,
            value=8.0,
            step=0.5
        )

    saved = (
        total_consumption * reduction / 100
    )

    remaining = (
        total_consumption - saved
    )

    estimated_savings = (
        saved * tariff
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Recorded Total",
        f"{total_consumption:,.2f}"
    )

    c2.metric(
        "Hypothetical Remaining",
        f"{remaining:,.2f}"
    )

    c3.metric(
        "Hypothetical Cost Difference",
        f"₹{estimated_savings:,.2f}"
    )

    comparison = pd.DataFrame({
        "Scenario": [
            "Original",
            "Hypothetical Optimized"
        ],
        "Consumption": [
            total_consumption,
            remaining
        ]
    })

    fig = px.bar(
        comparison,
        x="Scenario",
        y="Consumption",
        color="Scenario",
        title="What-If Consumption Comparison",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("🔮 Tomorrow's Predicted Demand")

    prediction, _ = forecast_next_day(
        daily
    )

    if prediction is None:

        st.warning(
            "Not enough data for the next-day forecast."
        )

    else:

        predicted_saving = (
            prediction * reduction / 100
        )

        predicted_remaining = (
            prediction - predicted_saving
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Demand",
            f"{prediction:,.3f}"
        )

        c2.metric(
            "After Hypothetical Reduction",
            f"{predicted_remaining:,.3f}"
        )

        c3.metric(
            "Potential Reduction",
            f"{predicted_saving:,.3f}"
        )

        st.caption(
            "This scenario applies your chosen reduction "
            "percentage to the forecast. It is not a measured "
            "or guaranteed saving."
        )

    st.divider()

    st.subheader("💡 Practical Recommendations")

    st.markdown("""
    **1. Review high-consumption periods**

    Investigate repeated peaks and compare them with
    operating schedules.

    **2. Review after-hours operation**

    Check whether nonessential equipment is running
    outside expected operating hours.

    **3. Investigate unusual readings**

    Compare flagged dates with maintenance, weather,
    events, and equipment schedules.

    **4. Consider flexible-load scheduling**

    Where safe and practical, evaluate moving flexible
    activities away from peak periods.

    **5. Preserve essential services**

    Never recommend shutting down essential equipment
    based only on low activity or energy readings.
    """)



# =====================================================
# PAGE 6: ENERGY ACTION PLANNER
# =====================================================

elif page == "Energy Action Planner":

    st.header("⚡ Energy Action Planner")

    st.write(
        "Turn unusual consumption patterns into "
        "a simple investigation plan."
    )

    st.info(
        "This tool prioritizes dates for investigation. "
        "It does not confirm energy waste or identify "
        "the exact cause."
    )

    # ---------------------------------------------
    # STEP 1: Find unusual high-consumption days
    # ---------------------------------------------

    st.subheader("1. Identify Days to Investigate")

    rolling_mean = daily.rolling(
        window=7,
        min_periods=3
    ).mean()

    rolling_std = daily.rolling(
        window=7,
        min_periods=3
    ).std()

    threshold = rolling_mean + 2 * rolling_std

    flagged_days = daily[
        (daily > threshold)
        & threshold.notna()
    ]

    if flagged_days.empty:

        st.success(
            "No unusually high days were flagged "
            "by the current threshold."
        )

    else:

        # -----------------------------------------
        # STEP 2: Calculate excess vs recent average
        # -----------------------------------------

        records = []

        for date in flagged_days.index:

            previous_days = daily[
                daily.index < date
            ].tail(7)

            if len(previous_days) == 0:
                continue

            previous_average = previous_days.mean()
            actual = daily.loc[date]

            excess = max(
                0,
                actual - previous_average
            )

            records.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Recorded Consumption": actual,
                "Previous 7-Day Average": previous_average,
                "Excess vs Recent Average": excess
            })

        action_table = pd.DataFrame(records)

        if action_table.empty:

            st.warning(
                "Not enough earlier data to build "
                "an investigation list."
            )

        else:

            # Largest excess first
            action_table = action_table.sort_values(
                "Excess vs Recent Average",
                ascending=False
            ).reset_index(drop=True)

            action_table.insert(
                0,
                "Priority",
                range(1, len(action_table) + 1)
            )

            st.caption(
                "Priority is based on excess above the "
                "previous 7 available days' average. "
                "It is not a confirmed waste ranking."
            )

            show_table(action_table)

            # -------------------------------------
            # STEP 3: Select a date
            # -------------------------------------

            st.divider()

            st.subheader("2. Investigate a Selected Day")

            selected_date = st.selectbox(
                "Choose a date",
                action_table["Date"].tolist()
            )

            selected_row = action_table[
                action_table["Date"] == selected_date
            ].iloc[0]

            st.metric(
                "Recorded Consumption",
                f"{selected_row['Recorded Consumption']:,.3f}"
            )

            st.metric(
                "Excess vs Recent Average",
                f"{selected_row['Excess vs Recent Average']:,.3f}"
            )

            # -------------------------------------
            # STEP 4: Compare with recent days
            # -------------------------------------

            selected_timestamp = pd.Timestamp(
                selected_date
            )

            comparison_days = daily[
                daily.index < selected_timestamp
            ].tail(7)

            comparison = pd.concat([
                comparison_days,
                daily.loc[[selected_timestamp]]
            ])

            comparison_df = comparison.rename(
                "Consumption"
            ).reset_index()

            comparison_df["Period"] = (
                comparison_df["timestamp"].dt.strftime(
                    "%Y-%m-%d"
                )
            )

            fig = px.bar(
                comparison_df,
                x="Period",
                y="Consumption",
                title="Selected Day vs Previous Available Days",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -------------------------------------
            # STEP 5: Investigation checklist
            # -------------------------------------

            st.divider()

            st.subheader("3. Investigation Checklist")

            st.write(
                "Use these checks to help identify "
                "possible reasons for the unusual usage."
            )

            check1 = st.checkbox(
                "Check operating schedules for this date"
            )

            check2 = st.checkbox(
                "Review maintenance or equipment records"
            )

            check3 = st.checkbox(
                "Check weather or special events"
            )

            check4 = st.checkbox(
                "Compare readings with similar weekdays"
            )

            check5 = st.checkbox(
                "Ask the facility team to verify the readings"
            )

            completed = sum([
                check1,
                check2,
                check3,
                check4,
                check5
            ])

            st.progress(completed / 5)

            st.caption(
                f"{completed} of 5 investigation checks completed."
            )

            if completed == 5:
                st.success(
                    "Checklist completed. Review the findings "
                    "with the facility team before taking action."
                )

            st.warning(
                "Do not switch off essential equipment based "
                "only on an unusual consumption reading."
            )





   

    import time

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        st.error(f"Person detection dependencies could not load: {e}")
        st.stop()

    st.title("Live Occupancy-Aware Energy Monitor")

    st.write(
        "Use your laptop webcam to detect people in the room "
        "and receive energy-saving recommendations."
    )

    st.info(
        "Alert-only demo: this app does not control lights, "
        "fans, or any electrical devices. A person outside "
        "the camera view may not be detected."
    )

    # -----------------------------
    # Settings
    # -----------------------------
    no_person_seconds = st.slider(
        "Alert if no person is detected for (seconds)",
        min_value=5,
        max_value=120,
        value=15,
        
elif page == "Occupancy-Aware Energy Control":
    import time
    import cv2
    import av
    import streamlit as st
    from ultralytics import YOLO
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        WebRtcMode,
    )

    st.title("Live Occupancy-Aware Energy Monitor")

    st.write(
        "Use your browser webcam to detect people and receive "
        "energy-saving recommendations."
    )

    st.info(
        "Alert-only demo. No lights, fans, or electrical "
        "devices are controlled automatically."
    )

    no_person_seconds = st.slider(
        "Alert if no person is detected for (seconds)",
        min_value=5,
        max_value=120,
        value=15,
        step=5,
    )

    confidence = st.slider(
        "Person detection confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
    )

    st.caption(
        "Higher confidence may reduce false detections, "
        "but can also miss people."
    )

    @st.cache_resource
    def load_person_model():
        return YOLO("yolov8n.pt")

    class PersonDetector(VideoProcessorBase):
        def __init__(self, model, conf, timeout):
            self.model = model
            self.conf = conf
            self.timeout = timeout
            self.no_person_since = None

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")

            results = self.model.predict(
                image,
                classes=[0],  # COCO class 0 = person
                conf=self.conf,
                imgsz=640,
                verbose=False,
            )

            person_count = 0

            for result in results:
                for box in result.boxes:
                    person_count += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 200, 0),
                        2,
                    )

                    cv2.putText(
                        image,
                        "Person",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 0),
                        2,
                    )

            now = time.time()

            if person_count > 0:
                self.no_person_since = None
                status = f"People detected: {person_count}"
                color = (0, 200, 0)
            else:
                if self.no_person_since is None:
                    self.no_person_since = now

                elapsed = now - self.no_person_since

                if elapsed >= self.timeout:
                    status = (
                        f"ALERT: No person detected for "
                        f"{int(elapsed)} seconds"
                    )
                    color = (0, 0, 255)
                else:
                    remaining = max(
                        0, int(self.timeout - elapsed)
                    )
                    status = (
                        f"No person detected - alert in "
                        f"{remaining}s"
                    )
                    color = (0, 165, 255)

            cv2.rectangle(
                image, (0, 0), (image.shape[1], 45),
                (25, 25, 25), -1
            )
            cv2.putText(
                image,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            return av.VideoFrame.from_ndarray(
                image, format="bgr24"
            )

    st.subheader("Browser Webcam")

    st.write(
        "Select START to allow camera access. Your browser "
        "may ask you to grant permission."
    )

    try:
        with st.spinner("Loading YOLO person detection model..."):
            model = load_person_model()

        webrtc_streamer(
            key="occupancy-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: PersonDetector(
                model, confidence, no_person_seconds
            ),
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            async_processing=True,
        )

        st.caption(
            "Green boxes indicate detected people. The video "
            "overlay shows the occupancy status and alert timer."
        )

    except Exception as e:
        st.error(
            "Could not initialize webcam detection. "
            f"Details: {e}"
        )

    st.divider()

    st.subheader("Energy-saving recommendation")

    st.write(
        "If the room appears unoccupied, manually check "
        "whether non-essential lights or equipment can be "
        "turned off. Never switch off essential equipment "
        "based only on camera detection."
    )

    st.caption(
        "Detection may fail because of poor lighting, "
        "occlusion, camera position, or network restrictions. "
        "No actual energy consumption or savings are measured."
    )


elif page == "Wi-Fi Intelligence":
    import pandas as pd

    st.title("Wi-Fi Intelligence & Usage Monitor")

    st.write(
        "Analyze Wi-Fi survey measurements, identify low "
        "throughput or low device-count readings, and provide "
        "recommendations for manual review."
    )

    st.info(
        "Alert-only prototype. This page does not disconnect "
        "Wi-Fi, change router settings, or control devices."
    )

    # --------------------------------
    # Threshold settings
    # --------------------------------
    st.subheader("Monitoring Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        min_throughput = st.number_input(
            "Minimum download speed (Mbps)",
            min_value=0.0,
            max_value=10000.0,
            value=15.0,
            step=1.0
        )

    with col2:
        min_devices = st.number_input(
            "Minimum device count",
            min_value=0,
            max_value=10000,
            value=5,
            step=1
        )

    # --------------------------------
    # Load dataset
    # --------------------------------
    @st.cache_data
    def load_wifi_data():
        return pd.read_csv("WiFiData.csv")

    try:
        df = load_wifi_data()

    except FileNotFoundError:
        st.error(
            "WiFiData.csv was not found. Place it in the "
            "same folder as app.py."
        )
        st.stop()

    # Ensure metrics are numeric
    df["SpeedDownload"] = pd.to_numeric(
        df["SpeedDownload"], errors="coerce"
    )

    df["SpeedUpload"] = pd.to_numeric(
        df["SpeedUpload"], errors="coerce"
    )

    df["NumberOfDevices"] = pd.to_numeric(
        df["NumberOfDevices"], errors="coerce"
    )

    # --------------------------------
    # Dataset overview
    # --------------------------------
    st.subheader("Dataset Overview")

    total_records = len(df)
    valid_download = df["SpeedDownload"].notna().sum()
    valid_devices = df["NumberOfDevices"].notna().sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total records", total_records)
    c2.metric("Valid speed readings", int(valid_download))
    c3.metric("Valid device counts", int(valid_devices))

    st.caption(
        "Missing device counts are excluded from device-count "
        "alerts; missing speed values are excluded from speed "
        "alerts."
    )

    # --------------------------------
    # Filter options
    # --------------------------------
    st.subheader("Filter by Location")

    locations = sorted(
        df["Location"].dropna().astype(str).unique().tolist()
    )

    selected_locations = st.multiselect(
        "Choose locations",
        options=locations,
        default=locations
    )

    filtered = df[
        df["Location"].astype(str).isin(selected_locations)
    ].copy()

    # --------------------------------
    # Flag unusual readings
    # --------------------------------
    filtered["Low Throughput"] = (
        filtered["SpeedDownload"].notna()
        & (filtered["SpeedDownload"] < min_throughput)
    )

    filtered["Low Device Count"] = (
        filtered["NumberOfDevices"].notna()
        & (filtered["NumberOfDevices"] < min_devices)
    )

    filtered["Review Needed"] = (
        filtered["Low Throughput"]
        | filtered["Low Device Count"]
    )

    st.divider()
    st.subheader("Network Metrics")

    valid_speeds = filtered["SpeedDownload"].dropna()
    valid_uploads = filtered["SpeedUpload"].dropna()
    valid_device_counts = filtered["NumberOfDevices"].dropna()

    a, b, c = st.columns(3)

    if not valid_speeds.empty:
        a.metric(
            "Average download",
            f"{valid_speeds.mean():.2f} Mbps"
        )
    else:
        a.metric("Average download", "N/A")

    if not valid_uploads.empty:
        b.metric(
            "Average upload",
            f"{valid_uploads.mean():.2f} Mbps"
        )
    else:
        b.metric("Average upload", "N/A")

    if not valid_device_counts.empty:
        c.metric(
            "Average device count",
            f"{valid_device_counts.mean():.1f}"
        )
    else:
        c.metric("Average device count", "N/A")

    # --------------------------------
    # Charts
    # --------------------------------
    st.subheader("Download Speed by Record")

    if not valid_speeds.empty:
        st.line_chart(
            filtered["SpeedDownload"].reset_index(drop=True)
        )
    else:
        st.warning("No valid download-speed readings.")

    st.subheader("Device Count by Record")

    if not valid_device_counts.empty:
        st.line_chart(
            filtered["NumberOfDevices"].reset_index(drop=True)
        )
    else:
        st.warning("No valid device-count readings.")

    # --------------------------------
    # Alerts
    # --------------------------------
    st.divider()
    st.subheader("Network Alerts")

    low_speed = filtered[filtered["Low Throughput"]]
    low_devices = filtered[filtered["Low Device Count"]]
    review_rows = filtered[filtered["Review Needed"]]

    x, y, z = st.columns(3)

    x.metric("Low-speed readings", len(low_speed))
    y.metric("Low-device readings", len(low_devices))
    z.metric("Records needing review", len(review_rows))

    if not review_rows.empty:
        st.warning(
            "Some measurements are below your configured "
            "thresholds. Review the records and network context."
        )

        display_cols = [
            col for col in [
                "Date",
                "Time",
                "Location",
                "SpeedDownload",
                "SpeedUpload",
                "NumberOfDevices",
                "Low Throughput",
                "Low Device Count"
            ]
            if col in review_rows.columns
        ]

        st.dataframe(
            review_rows[display_cols],
            use_container_width=True
        )

    else:
        st.success(
            "No valid readings fell below the configured "
            "thresholds in the selected locations."
        )

    # --------------------------------
    # Recommendations
    # --------------------------------
    st.divider()
    st.subheader("Recommended Actions")

    if not low_speed.empty:
        st.warning(
            "Low download speed detected in some records. "
            "Check signal strength, interference, router load, "
            "and whether the reading was temporary."
        )

    if not low_devices.empty:
        st.info(
            "Low device count detected in some records. "
            "Check whether the location was expected to be "
            "quiet before taking any action."
        )

    if review_rows.empty:
        st.write(
            "No threshold-based recommendation is needed "
            "for the selected records."
        )

    st.checkbox(
        "I reviewed the network readings and recommendations.",
        key="wifi_review_confirmed"
    )

    st.caption(
        "Important: Low throughput does not mean Wi-Fi should "
        "be turned off. It may indicate a connectivity problem. "
        "Device count alone does not establish that a network "
        "is unnecessary. This page analyzes historical CSV "
        "records; it is not a live router monitor."
    )
# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Energy Intelligence Prototype | "
    "Energy Analytics • Forecasting • Anomaly Flags • "
    "What-If Optimization"
)