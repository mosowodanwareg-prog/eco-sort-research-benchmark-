
import json
import time
import hashlib
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from google import genai
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

APP_VERSION = "2.0.0"

LABELS = [
    "Recyclable",
    "Non-recyclable",
    "Special disposal",
]

SPECIAL_LABEL = "Special disposal"

# Starter examples only. Verify reference labels before
# using the dataset for scientific conclusions.
STARTER_DATA = [
    ("Clean aluminum can", "Recyclable"),
    ("Clean glass bottle", "Recyclable"),
    ("Clean cardboard", "Recyclable"),
    ("Clean office paper", "Recyclable"),
    ("Clean steel food can", "Recyclable"),
    ("Clean PET drink bottle", "Recyclable"),
    ("Used household battery", "Special disposal"),
    ("Damaged lithium-ion battery", "Special disposal"),
    ("Old mobile phone", "Special disposal"),
    ("Fluorescent tube", "Special disposal"),
    ("Used motor oil", "Special disposal"),
    ("Uncured chemical paint waste", "Special disposal"),
    ("Greasy pizza box", "Non-recyclable"),
    ("Used paper tissue", "Non-recyclable"),
    ("Mixed-material snack wrapper", "Non-recyclable"),
    ("Food-soiled disposable container", "Non-recyclable"),
]

st.set_page_config(
    page_title="Eco-Sort Research Lab",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Eco-Sort Research Lab")
st.caption(f"Research benchmark • Version {APP_VERSION}")

st.markdown("""
### Research question
How do AI models differ in waste-classification performance,
special-disposal detection, and response latency?

This prototype compares models using the same test items and
reports their predictions, evaluation metrics, and errors.
""")

st.warning(
    "The starter dataset is small and its reference labels are preliminary. "
    "Verify labels against reliable sources and local disposal rules "
    "before making research claims."
)

# Secure configuration: use deployment secrets when available.
def read_secret():
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        return ""

api_key = read_secret()

if api_key:
    st.sidebar.success("API key loaded from application secrets.")
else:
    api_key = st.sidebar.text_input(
        "Google AI API key",
        type="password",
        help="Enter privately. Never publish your API key in GitHub."
    )

st.sidebar.header("Experiment settings")
st.sidebar.write("Temperature: 0")
st.sidebar.write("Identical test items for each model")

starter_df = pd.DataFrame(
    STARTER_DATA,
    columns=["item", "reference_label"],
)

st.header("1. Test dataset")

uploaded = st.file_uploader(
    "Upload a verified CSV dataset (optional)",
    type=["csv"],
    help="Required columns: item, reference_label",
)

dataset = starter_df.copy()

if uploaded is not None:
    try:
        candidate = pd.read_csv(uploaded)

        if not {"item", "reference_label"}.issubset(candidate.columns):
            st.error("CSV must contain item and reference_label columns.")
        else:
            candidate = candidate[["item", "reference_label"]].copy()
            candidate["item"] = candidate["item"].astype(str).str.strip()
            candidate["reference_label"] = (
                candidate["reference_label"].astype(str).str.strip()
            )

            valid = (
                candidate["item"].ne("")
                & candidate["reference_label"].isin(LABELS)
            )

            if candidate.empty or not valid.all():
                st.error(
                    "Fix empty items or invalid labels before using this CSV. "
                    "Allowed labels: " + ", ".join(LABELS)
                )
            else:
                dataset = candidate

    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")

dataset = dataset.reset_index(drop=True)

dataset_hash = hashlib.sha256(
    dataset.to_csv(index=False).encode("utf-8")
).hexdigest()

a, b, c = st.columns(3)
a.metric("Test items", len(dataset))
b.metric("Reference classes", dataset["reference_label"].nunique())
c.metric("Dataset fingerprint", dataset_hash[:10])

st.dataframe(dataset, use_container_width=True, hide_index=True)

st.download_button(
    "Download test dataset",
    data=dataset.to_csv(index=False).encode("utf-8"),
    file_name="eco_sort_dataset.csv",
    mime="text/csv",
)

st.header("2. Configure models")

left, right = st.columns(2)

with left:
    model_a = st.text_input("Model A ID", "gemini-2.5-flash")

with right:
    model_b = st.text_input("Model B ID", "gemini-2.5-pro")

st.caption(
    "Model availability depends on your API access. Verify both model IDs "
    "before running the experiment."
)

PROMPT = """
Classify the supplied waste item into exactly one category:
- Recyclable
- Non-recyclable
- Special disposal

Use Special disposal for items that may need dedicated collection
or careful handling. Do not assume that rules are identical worldwide.
This is a research prediction, not official local disposal advice.

Return only a JSON object with:
{
  "category": "one allowed category",
  "reason": "brief explanation"
}
"""

def predict(client, model_id, item):
    start = time.perf_counter()

    response = client.models.generate_content(
        model=model_id,
        contents=f"{PROMPT}\n\nItem: {item}",
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    latency = time.perf_counter() - start

    if not response.text:
        raise ValueError("The model returned an empty response.")

    parsed = json.loads(response.text)
    category = parsed.get("category")

    if category not in LABELS:
        raise ValueError(f"Unsupported category: {category!r}")

    return category, str(parsed.get("reason", "")), latency


def evaluate(frame):
    valid = frame[frame["prediction"].isin(LABELS)].copy()
    failed = len(frame) - len(valid)

    if valid.empty:
        return {
            "Test items": len(frame),
            "Valid predictions": 0,
            "Failed requests": failed,
            "Accuracy": None,
            "Macro precision": None,
            "Macro recall": None,
            "Macro F1": None,
            "Special-disposal recall": None,
            "Mean latency (seconds)": None,
        }

    y_true = valid["reference_label"]
    y_pred = valid["prediction"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    special_true = (y_true == SPECIAL_LABEL).astype(int)
    special_pred = (y_pred == SPECIAL_LABEL).astype(int)

    special_tp = int(((special_true == 1) & (special_pred == 1)).sum())
    special_fn = int(((special_true == 1) & (special_pred == 0)).sum())

    special_recall = (
        special_tp / (special_tp + special_fn)
        if special_tp + special_fn
        else 0.0
    )

    return {
        "Test items": len(frame),
        "Valid predictions": len(valid),
        "Failed requests": failed,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Macro precision": round(float(precision.mean()), 4),
        "Macro recall": round(float(recall.mean()), 4),
        "Macro F1": round(float(f1.mean()), 4),
        "Special-disposal recall": round(special_recall, 4),
        "Mean latency (seconds)": round(
            float(valid["latency_seconds"].mean()), 4
        ),
    }


def run_model(client, model_id, test_data):
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for index, row in enumerate(test_data.itertuples(index=False)):
        status.write(
            f"Testing {model_id}: {index + 1}/{len(test_data)}"
        )

        try:
            prediction, reason, latency = predict(
                client, model_id, row.item
            )

            rows.append({
                "model": model_id,
                "item": row.item,
                "reference_label": row.reference_label,
                "prediction": prediction,
                "correct": prediction == row.reference_label,
                "latency_seconds": round(latency, 4),
                "reason": reason,
                "error": "",
            })

        except Exception as exc:
            rows.append({
                "model": model_id,
                "item": row.item,
                "reference_label": row.reference_label,
                "prediction": "FAILED",
                "correct": None,
                "latency_seconds": None,
                "reason": "",
                "error": str(exc),
            })

        progress.progress((index + 1) / len(test_data))

    progress.empty()
    status.empty()

    return pd.DataFrame(rows)


st.header("3. Run experiment")

st.info(
    "API usage may incur charges or quotas. Check your provider's terms "
    "before running the benchmark."
)

if st.button("Run benchmark", type="primary", use_container_width=True):
    if not api_key.strip():
        st.error("Enter your API key in the sidebar first.")
    elif not model_a.strip() or not model_b.strip():
        st.error("Enter both model IDs.")
    elif dataset.empty:
        st.error("The test dataset is empty.")
    else:
        try:
            client = genai.Client(api_key=api_key.strip())
            model_ids = list(dict.fromkeys([
                model_a.strip(),
                model_b.strip(),
            ]))

            all_results = []

            for model_id in model_ids:
                st.subheader(f"Evaluating {model_id}")
                all_results.append(run_model(client, model_id, dataset))

            results = pd.concat(all_results, ignore_index=True)

            run_metadata = {
                "app_version": APP_VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": dataset_hash,
                "models": model_ids,
                "temperature": 0,
                "test_items": len(dataset),
                "reference_labels_verified": False,
            }

            st.session_state["eco_results"] = results
            st.session_state["eco_metadata"] = run_metadata

        except Exception as exc:
            st.error("Experiment setup failed. Check your key and model access.")
            st.caption(str(exc))


if "eco_results" in st.session_state:
    results = st.session_state["eco_results"]
    metadata = st.session_state["eco_metadata"]

    st.header("4. Results and error analysis")

    st.subheader("Run metadata")
    st.json(metadata)

    summaries = []

    for model_id in results["model"].drop_duplicates():
        subset = results[results["model"] == model_id]
        summaries.append({
            "Model": model_id,
            **evaluate(subset),
        })

    st.subheader("Overall metrics")
    st.dataframe(
        pd.DataFrame(summaries),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Per-class performance")
    per_class = []

    for model_id in results["model"].drop_duplicates():
        subset = results[
            (results["model"] == model_id)
            & results["prediction"].isin(LABELS)
        ]

        if subset.empty:
            continue

        p, r, f, support = precision_recall_fscore_support(
            subset["reference_label"],
            subset["prediction"],
            labels=LABELS,
            zero_division=0,
        )

        for index, label in enumerate(LABELS):
            per_class.append({
                "Model": model_id,
                "Class": label,
                "Precision": round(float(p[index]), 4),
                "Recall": round(float(r[index]), 4),
                "F1": round(float(f[index]), 4),
                "Support": int(support[index]),
            })

    st.dataframe(
        pd.DataFrame(per_class),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Confusion matrices")

    for model_id in results["model"].drop_duplicates():
        subset = results[
            (results["model"] == model_id)
            & results["prediction"].isin(LABELS)
        ]

        if subset.empty:
            st.write(f"No valid predictions for {model_id}.")
            continue

        matrix = confusion_matrix(
            subset["reference_label"],
            subset["prediction"],
            labels=LABELS,
        )

        st.write(f"Model: {model_id} (rows = reference labels)")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=LABELS,
                columns=LABELS,
            ),
            use_container_width=True,
        )

    st.subheader("Incorrect predictions and failed requests")

    errors = results[
        (results["correct"] == False)
        | (results["prediction"] == "FAILED")
    ]

    if errors.empty:
        st.success("No errors were recorded in this run.")
    else:
        st.dataframe(errors, use_container_width=True, hide_index=True)

    st.download_button(
        "Download predictions (CSV)",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="eco_sort_predictions.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download run metadata (JSON)",
        data=json.dumps(metadata, indent=2).encode("utf-8"),
        file_name="eco_sort_run_metadata.json",
        mime="application/json",
    )

st.divider()

st.caption(
    "Research integrity: results describe only the tested items, prompts, "
    "model versions, and API configuration. The starter labels are not "
    "universally authoritative. Verify labels and report limitations."
)
