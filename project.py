import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="📱 Mobile User Segmentation",
    page_icon="📱",
    layout="wide"
)

# ---------------------------------
# DARK THEME CSS
# ---------------------------------
st.markdown("""
<style>

.stApp{
background:#0f172a;
color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
background:#111827;
}

/* Headings */
h1,h2,h3,h4,h5,h6{
color:#38bdf8;
}

/* Paragraph */
p,label{
color:white;
}

/* Metrics */
[data-testid="stMetric"]{
background:#1e293b;
padding:15px;
border-radius:12px;
border:1px solid #334155;
}

[data-testid="stMetricValue"]{
color:#38bdf8;
font-size:28px;
font-weight:bold;
}

/* Buttons */
div.stButton > button{
background:#2563eb;
color:white;
border:none;
border-radius:10px;
height:45px;
font-size:18px;
width:100%;
transition:0.3s;
}

div.stButton > button:hover{
background:#1d4ed8;
transform:scale(1.02);
}

/* DataFrame */
[data-testid="stDataFrame"]{
border-radius:12px;
overflow:hidden;
}

/* Success Box */
.stAlert{
border-radius:10px;
}

hr{
border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>📱</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;'>Mobile User</h2>",
        unsafe_allow_html=True
    )

    st.caption("Machine Learning Dashboard")

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Dashboard",
            "Dataset",
            "Statistics",
            "Visualization",
            "Prediction"
        ],
        icons=[
            "house",
            "table",
            "bar-chart",
            "pie-chart",
            "cpu"
        ],
        default_index=0
    )

    st.divider()

    st.success("🟢 Model Ready")

# ---------------------------------
# LOAD DATA
# ---------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("user.csv")

df = load_data()

# ---------------------------------
# FEATURES
# ---------------------------------

x = df[
    [
        "Screen_Time",
        "Data_Usage",
        "Recharge_Amount"
    ]
]

# ---------------------------------
# MODEL
# ---------------------------------

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(x)

# ===========================================
# DASHBOARD
# ===========================================

if selected == "Dashboard":

    st.title("📱 Mobile User Segmentation System")

    st.caption("K-Means Clustering | Unsupervised Learning")

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric(
            "👥 Total Users",
            len(df)
        )

    with c2:
        st.metric(
            "📊 Clusters",
            3
        )

    with c3:
        st.metric(
            "🤖 Algorithm",
            "K-Means"
        )

    with c4:
        st.metric(
            "📌 Features",
            x.shape[1]
        )

    st.divider()

    left,right = st.columns([2,1])

    with left:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )

    with right:

        st.subheader("📈 Quick Statistics")

        st.metric(
            "Average Screen Time",
            f"{df['Screen_Time'].mean():.1f} Hours"
        )

        st.metric(
            "Average Data Usage",
            f"{df['Data_Usage'].mean():.1f} GB"
        )

        st.metric(
            "Average Recharge",
            f"₹{df['Recharge_Amount'].mean():.0f}"
        )

        st.progress(100)

        st.success("Dataset Loaded Successfully")

# ===========================================
# DATASET PAGE
# ===========================================

elif selected == "Dataset":

    st.title("📂 Dataset Explorer")

    st.caption("Explore Mobile User Dataset")

    st.divider()

    search = st.text_input(
        "🔍 Search User"
    )

    if search:

        filtered = df[
            df.astype(str)
            .apply(lambda x: x.str.contains(search, case=False))
            .any(axis=1)
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            height=500
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

    st.download_button(
        "⬇ Download Dataset",
        df.to_csv(index=False),
        "mobile_users.csv",
        "text/csv"
    )
    # ==========================================
# STATISTICS
# ==========================================

elif selected == "Statistics":

    st.title("📊 Dataset Statistics")
    st.caption("Complete Analysis of Mobile User Dataset")

    tab1, tab2, tab3 = st.tabs(
        ["📄 Summary", "❌ Missing Values", "📊 Cluster Count"]
    )

    # --------------------------------------

    with tab1:

        st.subheader("Summary Statistics")

        st.dataframe(
            df.describe().T,
            use_container_width=True
        )

    # --------------------------------------

    with tab2:

        st.subheader("Missing Values")

        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            missing,
            use_container_width=True
        )

    # --------------------------------------

    with tab3:

        st.subheader("Cluster Distribution")

        st.bar_chart(df["Cluster"].value_counts())


# ==========================================
# VISUALIZATION
# ==========================================

elif selected == "Visualization":

    st.title("📈 Data Visualization")

    st.caption("Visual Analysis of Mobile Users")

    st.divider()

    col1, col2 = st.columns(2)

    # =====================================
    # Scatter Plot
    # =====================================

    with col1:

        st.subheader("📍 Screen Time vs Recharge")

        fig, ax = plt.subplots(figsize=(6,4))

        scatter = ax.scatter(
            df["Screen_Time"],
            df["Recharge_Amount"],
            c=df["Cluster"],
            s=80,
            alpha=0.8
        )

        ax.set_xlabel("Screen Time (Hours)")
        ax.set_ylabel("Recharge Amount (₹)")
        ax.set_title("Cluster Distribution")

        st.pyplot(fig)

    # =====================================
    # Pie Chart
    # =====================================

    with col2:

        st.subheader("🥧 Cluster Percentage")

        fig2, ax2 = plt.subplots(figsize=(5,5))

        ax2.pie(
            df["Cluster"].value_counts(),
            labels=[
                "Cluster 0",
                "Cluster 1",
                "Cluster 2"
            ],
            autopct="%1.1f%%",
            startangle=90
        )

        ax2.set_title("Cluster Distribution")

        st.pyplot(fig2)

    st.divider()

    # =====================================
    # Bar Chart
    # =====================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("📊 Average Recharge")

        recharge = (
            df.groupby("Cluster")["Recharge_Amount"]
            .mean()
        )

        st.bar_chart(recharge)

    # =====================================
    # Average Screen Time
    # =====================================

    with col4:

        st.subheader("📱 Average Screen Time")

        screen = (
            df.groupby("Cluster")["Screen_Time"]
            .mean()
        )

        st.line_chart(screen)

    st.divider()

    # =====================================
    # Correlation
    # =====================================

    st.subheader("📌 Feature Correlation")

    corr = x.corr()

    st.dataframe(
        corr.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    st.divider()

    # =====================================
    # Box Plot
    # =====================================

    st.subheader("📦 Recharge Amount Distribution")

    fig3, ax3 = plt.subplots(figsize=(8,4))

    ax3.boxplot(df["Recharge_Amount"])

    ax3.set_ylabel("Recharge Amount")

    st.pyplot(fig3)
    # ==========================================
# PREDICTION
# ==========================================

elif selected == "Prediction":

    st.title("🎯 Mobile User Prediction")
    st.caption("Predict User Category using K-Means Clustering")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        screen = st.slider(
            "📱 Screen Time (Hours)",
            1.0,
            12.0,
            5.0,
            0.5
        )

        data = st.number_input(
            "🌐 Data Usage (GB)",
            min_value=1,
            max_value=100,
            value=20
        )

    with col2:

        recharge = st.number_input(
            "💳 Recharge Amount (₹)",
            min_value=50,
            max_value=5000,
            value=399
        )

    st.divider()

    if st.button("🚀 Predict User Category", use_container_width=True):

        user = [[screen, data, recharge]]

        cluster = model.predict(user)[0]

        category = {
            0: "🔵 Light User",
            1: "🟢 Regular User",
            2: "🔥 Heavy User"
        }

        color = {
            0: "#2563eb",
            1: "#22c55e",
            2: "#ef4444"
        }

        st.markdown(
            f"""
<div style="
background:{color[cluster]};
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-size:28px;
font-weight:bold;">
Prediction Result<br>
{category[cluster]}
</div>
""",
            unsafe_allow_html=True
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("📌 Prediction Reason")

            if cluster == 0:
                st.success("""
✅ Low Screen Time

✅ Low Data Usage

✅ Low Recharge
""")

            elif cluster == 1:
                st.info("""
📱 Average Screen Time

🌐 Average Data Usage

💳 Average Recharge
""")

            else:
                st.error("""
🔥 High Screen Time

🌐 High Data Usage

💳 High Recharge
""")

        with right:

            st.subheader("📋 Entered Details")

            summary = pd.DataFrame({

                "Feature":[
                    "Screen Time",
                    "Data Usage",
                    "Recharge Amount"
                ],

                "Value":[
                    f"{screen} Hours",
                    f"{data} GB",
                    f"₹{recharge}"
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("📈 User Usage Meter")

        if cluster == 0:
            st.progress(30)

        elif cluster == 1:
            st.progress(65)

        else:
            st.progress(100)

        st.divider()

        csv = summary.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Report",
            data=csv,
            file_name="Prediction_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;padding:20px;">

<h2 style="color:#38bdf8;">
📱 Mobile User Segmentation System
</h2>

<h4 style="color:white;">
Developed by <span style="color:#22c55e;">Suhani Gupta</span>
</h4>

<p style="font-size:18px;">
Python Developer | AI & ML Enthusiast
</p>

<p>
Machine Learning Project using
<b>K-Means Clustering</b>
</p>

<p>
Guided by
<b>Chandrama Sir</b>
</p>

<hr>

<p style="font-size:15px;">
© 2026 All Rights Reserved
</p>

</div>
""",
unsafe_allow_html=True
)