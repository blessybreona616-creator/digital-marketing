import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Digital Marketing Conversion Prediction", page_icon="📊", layout="wide")

st.title("Digital Marketing Campaign Conversion Prediction")
st.write(
    "This application explores a digital marketing dataset, compares four machine-learning "
    "models, and predicts the conversion likelihood for a customer profile."
)

DATA_FILE = "marketing_data.csv"

@st.cache_data
def load_data_from_bytes(file_bytes):
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))

@st.cache_data
def load_local_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_models(data):
    # Follow the project notebook: remove the customer ID and target from input features.
    X = data.drop(columns=["CustomerID", "Conversion"], errors="ignore")
    # ConversionRate is excluded because it may not be available when making a prediction.
    X = X.drop(columns=["ConversionRate"], errors="ignore")
    y = data["Conversion"].astype(int)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    }

    trained = {}
    rows = []
    for name, estimator in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
        trained[name] = pipe

    metrics = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_name = metrics.loc[0, "Model"]
    return X, y, X_test, y_test, trained, metrics, best_name

uploaded_file = st.file_uploader(
    "Upload the project dataset (CSV)",
    type=["csv"],
    help="Choose your marketing dataset. The file is processed by this app for the current session."
)

try:
    if uploaded_file is not None:
        df = load_data_from_bytes(uploaded_file.getvalue())
    elif __import__("os").path.exists(DATA_FILE):
        df = load_local_data()
    else:
        st.info("Upload your marketing_data.csv file to start the demonstration.")
        st.stop()
except Exception as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

required_columns = {"CustomerID", "Conversion"}
if not required_columns.issubset(df.columns):
    st.error("The dataset must contain the columns CustomerID and Conversion.")
    st.stop()

if not set(df["Conversion"].dropna().unique()).issubset({0, 1}):
    st.error("The Conversion target must contain only 0 and 1.")
    st.stop()

with st.spinner("Preparing the data and training the four models..."):
    X, y, X_test, y_test, trained_models, results_df, best_model_name = train_models(df)

st.success("Models trained successfully.")

tab1, tab2, tab3 = st.tabs(["Project overview", "Model comparison", "Try a prediction"])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customer records", f"{len(df):,}")
    c2.metric("Dataset columns", len(df.columns))
    c3.metric("Training / testing split", "80% / 20%")

    st.write("The target column is **Conversion**: 1 means converted and 0 means not converted.")
    counts = df["Conversion"].value_counts().reindex([0, 1], fill_value=0)
    chart_df = pd.DataFrame({
        "Conversion status": ["Not converted (0)", "Converted (1)"],
        "Number of customers": [int(counts.loc[0]), int(counts.loc[1])]
    })
    st.bar_chart(chart_df.set_index("Conversion status"))

    st.subheader("Preview of the dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Missing values")
    missing = df.isnull().sum().rename("Missing values").to_frame()
    st.dataframe(missing, use_container_width=True)

with tab2:
    st.subheader("Comparison of machine-learning models")
    st.caption("Metrics are calculated on the held-out 20% test set.")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1 Score": "{:.4f}", "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True
    )
    st.info(f"Best model by F1-score in this run: **{best_model_name}**")

    metric_choice = st.selectbox(
        "Choose a metric to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )
    plot_df = results_df.set_index("Model")[[metric_choice]]
    st.bar_chart(plot_df)

    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not converted", "Converted"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix – {best_model_name}")
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Enter a customer profile")
    st.write("Choose values for the available customer and campaign features, then select Predict.")
    best_model = trained_models[best_model_name]

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    with st.form("prediction_form"):
        left, right = st.columns(2)
        user_values = {}

        for index, column in enumerate(numeric_columns):
            target_col = left if index % 2 == 0 else right
            series = pd.to_numeric(X[column], errors="coerce").dropna()
            default = float(series.median()) if not series.empty else 0.0
            minimum = float(series.min()) if not series.empty else 0.0
            maximum = float(series.max()) if not series.empty else max(default, 1.0)
            if minimum == maximum:
                user_values[column] = target_col.number_input(
                    column, value=default, format="%.4f"
                )
            else:
                user_values[column] = target_col.number_input(
                    column, min_value=minimum, max_value=maximum,
                    value=min(max(default, minimum), maximum), format="%.4f"
                )

        for index, column in enumerate(categorical_columns):
            options = sorted(X[column].dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            target_col = left if index % 2 == 0 else right
            user_values[column] = target_col.selectbox(column, options=options)

        submitted = st.form_submit_button("Predict conversion", type="primary")

    if submitted:
        input_df = pd.DataFrame([user_values], columns=X.columns)
        prediction = int(best_model.predict(input_df)[0])
        probability = float(best_model.predict_proba(input_df)[0, 1])

        st.divider()
        st.subheader("Prediction result")
        st.metric("Estimated conversion probability", f"{probability:.1%}")
        if prediction == 1:
            st.success("The model predicts: **Likely to convert (1)**")
        else:
            st.warning("The model predicts: **Less likely to convert (0)**")
        st.caption(
            "This is a machine-learning estimate based on patterns in the supplied dataset; "
            "it is not a guarantee that a customer will purchase."
        )

st.divider()
st.caption("Student project: Digital Marketing Campaign Conversion Prediction")
