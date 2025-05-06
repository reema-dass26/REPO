import streamlit as st
import os
import pandas as pd
import json
import plotly.graph_objects as go
import ast
from math import ceil
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
from pprint import pprint
from streamlit_option_menu import option_menu
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import time
from datetime import datetime
import re

st.set_page_config(
    page_title="Building Bridges in Research: Integrating Provenance and Data Management in Virtual Research Environments",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)############################################################################
#Helper functions
############################################################################

def detect_deprecated_code(df: pd.DataFrame, deprecated_commits: List[str], **_) -> List[Dict[str, Any]]:
    commit_col = 'GIT_current_commit_hash'
    if commit_col not in df.columns:
        raise KeyError(f"Missing {commit_col} in DataFrame")
    out = df[df[commit_col].isin(deprecated_commits)]
    cols = ['run_id', commit_col, 'tag_notebook_name', 'tag_mlflow.runName']
    cols = [c for c in cols if c in df.columns]
    return out[cols].to_dict(orient='records')

def map_model_dataset(df: pd.DataFrame, **_) -> List[Dict[str, Any]]:
    model_col = 'tag_model_name' if 'tag_model_name' in df.columns else 'param_model_name'
    
    cols = [
        'run_id',
        model_col,
        'param_dataset.title',
        'param_dataset.doi',
        'param_dataset.published',
        'param_dataset.publisher'
    ]
    cols = [c for c in cols if c in df.columns]
    
    data = df[cols].to_dict(orient='records')
    
    # 🔥 Post-process: make DOI into clickable links
    for record in data:
        doi = record.get('param_dataset.doi')
        if doi:
            # If the DOI is already a URL, fine; else, prepend https://doi.org/
            doi_link = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
            record['param_dataset.doi'] = f"[{doi}]({doi_link})"
    
    return data

@st.cache_data
def load_data():
    """
    Load and flatten JSON metadata files from the MODEL_PROVENANCE directory.
    """
    files = glob.glob(os.path.join("MODEL_PROVENANCE", "*", "*_run_summary.json"))

    if not files:
        st.warning("⚠️ No run summary JSON files found inside MODEL_PROVENANCE!")
        return pd.DataFrame()

    rows = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
                summary = json.loads(content)

            row = {"run_id": summary.get("ML_EXP_run_id", summary.get("run_id", ""))}

            # ✅ Correctly load ML_EXP-prefixed metadata
            row.update({f"param_{k}": v for k, v in summary.get("ML_EXP_params", {}).items()})
            row.update({f"metric_{k}": v for k, v in summary.get("ML_EXP_metrics", {}).items()})
            row.update({k: v for k, v in summary.get("ML_EXP_tags", {}).items()})

            # ✅ Also add any top-level keys you care about
            for top_key in [
                "MLSEA_hyperparameters", "MLSEA_computeEnvironment", "MLSEA_dataPreprocessing",
                "MLSEA_modelArchitecture", "MLSEA_trainingProcedure", "MLSEA_evaluationMetrics",
                "ML_EXP_training_start_time", "ML_EXP_training_end_time"
            ]:
                if top_key in summary:
                    row[top_key] = summary[top_key]

            rows.append(row)

        except Exception as e:
            st.error(f"❌ Error loading file {file_path}: {e}")

    if not rows:
        st.warning("⚠️ No valid run summary data could be loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    st.success(f"✅ Loaded {len(df)} runs with {len(df.columns)} columns.")
    return df

import pandas as pd


def _get_all_features(df):
    """
    Retrieve the list of feature names from the DataFrame.
    Assumes every row has the same 'param_feature_names'.
    """
    raw = df.loc[0, 'param_feature_names']
    return ast.literal_eval(raw)

def evaluate_subset(features, test_size=0.2, random_state=42, n_estimators=200):
    """
    Train and evaluate a RandomForestClassifier on a subset of features from iris_data.json.
    """
    # 1. Load and parse the dataset
    with open("iris_data.json", "r") as f:
        dataset = json.load(f)

    df = pd.DataFrame(dataset)
    target_col = df.columns[-1]  # Assuming the last column is the label
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 2. Drop ID column if it exists
    id_cols = [c for c in X.columns if c.lower() == "id"]
    X = X.drop(columns=id_cols, errors="ignore")

    # 3. Coerce numeric columns
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            pass

    # 4. Label encode the target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 5. Use only the selected features
    X_sub = X[features]

    # 6. Train/test split and model evaluation
    Xtr, Xte, ytr, yte = train_test_split(X_sub, y, test_size=test_size, random_state=random_state)
    m = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    m.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))


def trace_preprocessing(df, run_id=None):
    """
    Extract preprocessing trace information for a given run_id or all runs.
    """
    cols = ['run_id',
            'param_dataset.title',
            'param_columns_raw',
            'param_dropped_columns',
            'param_feature_names',
            'param_dataset.authors', 'param_dataset.doi', 'param_dataset.published',
            'param_test_size',
            'param_criterion',
            'param_max_depth','param_max_leaf_nodes', 'param_max_samples',
           'metric_accuracy','metric_f1_macro','metric_roc_auc']
    if run_id is None:
        subset = df.loc[:, cols]
    else:
        subset = df.loc[df['run_id'] == run_id, cols]
    return subset.to_dict(orient='records')

def drop_impact(df, feature, **_):
    """
    Evaluate the impact of dropping a single feature on model accuracy.
    """
    all_feats = _get_all_features(df)
    baseline = evaluate_subset(all_feats)
    without = [f for f in all_feats if f != feature]
    dropped = evaluate_subset(without)
    return {
      'dropped_feature': feature,
      'baseline_acc': baseline,
      'dropped_acc': dropped,
      'impact': baseline - dropped
    }

def drop_impact_all(df):
    """
    Compute drop-impact for every feature in the dataset.
    Returns list of dicts with dropped_feature, baseline_acc, dropped_acc, impact.
    """
    feats = _get_all_features(df)
    baseline = evaluate_subset(feats)
    summary = []
    for feat in feats:
        without = [f for f in feats if f != feat]
        acc = evaluate_subset(without)
        summary.append({
            'dropped_feature': feat,
            'baseline_acc': baseline,
            'dropped_acc': acc,
            'impact': round(baseline - acc, 4)
        })
    return summary

def best_feature_subset(df, features, **_):
    """
    Evaluate model accuracy using a specified subset of features.
    """
    acc = evaluate_subset(features)
    return {'features': features, 'accuracy': acc}

def common_high_accuracy(df, threshold=0.95):
    """
    Filter runs with test accuracy >= threshold and list unique shared preprocessing settings.
    """
    high = df[df['metric_accuracy_score_X_test'] >= threshold]
    cols = ['param_dropped_columns', 'param_test_size', 'param_feature_names']
    return high[cols].drop_duplicates().to_dict(orient='records')

USE_CASES = {
    'trace_preprocessing': {
        'func': trace_preprocessing,
        'required_params': [],
        'optional_params': ['run_id'],
    },
    'drop_impact': {
        'func': drop_impact,
        'required_params': ['feature'],
        'optional_params': [],
    },
    'drop_impact_all': {
        'func': drop_impact_all,
        'required_params': [],
        'optional_params': [],
    },
    'best_feature_subset': {
        'func': best_feature_subset,
        'required_params': ['features'],
        'optional_params': [],
    },
    'common_high_accuracy': {
        'func': common_high_accuracy,
        'required_params': ['threshold'],
        'optional_params': [],
    },
}
# —— Find latest run summary with justification data ——

def get_latest_justification_summary(base_dir="MODEL_PROVENANCE"):
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    timestamped_folders = []
    for folder in folders:
        match = re.search(r'_(v\d{8}_\d{6})', folder)
        if match:
            try:
                timestamp = datetime.strptime(match.group(1), "v%Y%m%d_%H%M%S")
                timestamped_folders.append((timestamp, folder))
            except ValueError:
                continue

    if not timestamped_folders:
        raise FileNotFoundError("No timestamped folders found in MODEL_PROVENANCE")

    latest_folder = max(timestamped_folders)[1]
    file_path = os.path.join(base_dir, latest_folder, f"{latest_folder}_run_summary.json")
    return file_path

# —— Load justifications and return as DataFrame ——

def load_justification_table(path):
    import json
    import pandas as pd

    try:
        with open(path, "r") as f:
            js = json.load(f)
    except Exception as e:
        return pd.DataFrame([{"Decision": "Error", "Justification": f"Failed to load file: {e}"}])

    # Safely extract justifications from nested tags
    tags = js.get("ML_EXP_tags", {})
    justifications = {
        k: v for k, v in tags.items()
        if k.startswith("justification_") and isinstance(v, str)
    }

    # Fallback if none found
    if not justifications:
        return pd.DataFrame([{
            "Decision": "No justifications recorded",
            "Justification": "—"
        }])

    rows = [
        {
            "Decision": k.replace("justification_", "").replace("_", " ").capitalize(),
            "Justification": v.strip() if isinstance(v, str) else str(v)
        }
        for k, v in justifications.items()
    ]

    return pd.DataFrame(rows)


df = load_data()
USE_CASES = {
    'trace_preprocessing': {
        'func': trace_preprocessing,
        'required_params': [],
        'optional_params': ['run_id'],
    },
    'drop_impact': {
        'func': drop_impact,
        'required_params': ['feature'],
        'optional_params': [],
    },
    'drop_impact_all': {
        'func': drop_impact_all,
        'required_params': [],
        'optional_params': [],
    },
    'best_feature_subset': {
        'func': best_feature_subset,
        'required_params': ['features'],
        'optional_params': [],
    },
    'common_high_accuracy': {
        'func': common_high_accuracy,
        'required_params': ['threshold'],
        'optional_params': [],
    },
     'detect_deprecated_code': {
        'func': detect_deprecated_code,
        'required_params': ['deprecated_commits'],
        'optional_params': []
    },
    'map_model_dataset': {
        'func': map_model_dataset,
        'required_params': [],
        'optional_params': []
    },
}

with st.sidebar:
    selected = option_menu(
        menu_title="📂 Navigation",
        options=[
            "🏠 Dashboard",
            "📁 Dataset Metadata",
            "🧠 ML Model Metadata",
            "📊 Model Plots",
            "🛰️ Provenance Trace",
            "🧨 Error & Version Impact",
            "🧭 Model-Dataset Mapping",
            "📣 Notify Outdated Forks",
            "📤 Export Provenance",
            "📘 Researcher Justifications",
            "📚 Invenio Metadata",
            # "⚠️ Deprecated Code Check",

            

        ],
        icons=[
            "house", "database", "gear", "bar-chart", "globe", "link", "exclamation-triangle","map", "megaphone" , "book","cloud-download"
        ],
        menu_icon="cast",
        default_index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; font-size: 13px; color: gray;'>"
        "🚀 Designed with ❤️ by <strong>Reema Dass</strong>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='
            font-weight: bold;
            color: #ff4b4b;
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 5px;
        '>🎯 Infra Flow</div>
        """,
        unsafe_allow_html=True
    )

    infra_nodes = [
        Node(id="DBRepo", label="DBRepo 📚", color="#f94144"),
        Node(id="Invenio", label="Invenio 💃", color="#f3722c"),
        Node(id="JupyterHub", label="Jupyter 💻", color="#f8961e"),
        Node(id="GitHub", label="GitHub 🧠", color="#f9844a"),
        Node(id="VRE", label="VRE 🧪", color="#43aa8b"),
        Node(id="Metadata", label="Metadata 🧰", color="#577590"),
        Node(id="Provenance JSON", label="JSON 📜", color="#277da1"),
        Node(id="Visualization", label="Viz 🌐", color="#9b5de5")
    ]

    infra_edges = [
        Edge(source="DBRepo", target="VRE"),
        Edge(source="Invenio", target="VRE"),
        Edge(source="JupyterHub", target="VRE"),
        Edge(source="GitHub", target="VRE"),
        Edge(source="Metadata", target="Provenance JSON"),
        Edge(source="Provenance JSON", target="Visualization"),
        Edge(source="VRE", target="Visualization")
    ]

    node_count = len(infra_nodes)
    graph_height = max(300, ceil(node_count * 80))
    
    graph_config = Config(
    width=250,  # slightly wider than before
    height=graph_height,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#FFDD00",
    collapsible=True,
    node={'labelProperty': 'label'},
    link={'renderLabel': False},
    fontColor="#000000"
)

    
    agraph(nodes=infra_nodes, edges=infra_edges, config=graph_config)


# Header
st.markdown("<h1 style='text-align: center;'>Building Bridges in Research: Integrating Provenance and Data Management in Virtual Research Environments</h1>", unsafe_allow_html=True)

# Main content switching


if selected == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    
    # Center content using columns
    left_col, center_col, right_col = st.columns([1, 5, 1])  # Adjust ratios as needed

    with center_col:
        st.markdown("## 👋 Welcome to the End-to-End Provenance Dashboard")

        st.markdown("""
Welcome to your one-stop hub for managing and understanding the **provenance** of machine learning experiments within Virtual Research Environments (VREs).  
This dashboard empowers **traceability**, **reproducibility**, and **transparency** across your ML lifecycle.

---

### 🧩 Key Features

#### 🧬 Dataset Metadata
- View dataset authorship, publication, versioning  
- Integrated with source platforms, DOIs, and database records

#### 🧠 ML Model Metadata
- Inspect training configurations, sampling techniques, hyperparameters  
- Runtime environment details (Python, libraries, hardware)

#### 📊 Model Plots
- Feature importance, confusion matrix, ROC & PR curves  
- SHAP explanations, linked with model/dataset metadata

#### 🛰️ Provenance Trace
- Track how specific outputs were produced  
- Compare configurations, parameters, and preprocessing steps

#### ⚠️ Error & Version Impact
- Flag outdated experiments using old Git commits or deprecated data/code versions

#### 🧭 Model–Dataset Mapping
- Visualize relationships between trained models and datasets  
- DOI, publisher, and attribution metadata included

#### 📣 Notify Outdated Forks
- Automatically notify contributors using outdated forks via GitHub Issues

#### 📤 Export Provenance
- Export training and provenance metadata in standardized formats (YAML, JSON, etc.)

#### 📘 Researcher Justifications
- Capture rationales for decisions taken during the ML pipeline

#### 📚 Invenio Metadata
- Integrate with Invenio-style metadata records for publications and datasets

---



---

🔍 Use the **sidebar navigation** to explore each section. This tool is particularly useful for:
- Reproducibility validation
- Research auditing
- FAIR data practices
- Collaborative ML workflows

""")
        
    st.markdown("---")
    st.markdown("---")
    st.markdown("## 🔄 ML Infrastructure Flow: Visual + Narrative")
    
    col1, col2 = st.columns([1, 1.4])
    
    with col1:
        if st.button("▶️ Start Flow"):
            st.markdown("### 🔍 Narrative Walkthrough")
            st.markdown("**📦 DBRepo** — provides structured datasets to power experiments")
            time.sleep(1)
            st.markdown("**💻 JupyterHub** — where ML code is developed and run")
            time.sleep(1)
            st.markdown("**🧠 GitHub** — version control for all notebooks & code")
            time.sleep(1)
            st.markdown("**🗃️ Invenio** — stores trained models, logs, and artifacts")
            time.sleep(1)
            st.markdown("**🧪 VRE (Virtual Research Environment)** — a unified system connecting code, data, compute, and storage")
            time.sleep(1)
            st.markdown("**🧰 Metadata Extractor** — pulls details from each component to track provenance")
            time.sleep(1)
            st.markdown("**📜 Provenance JSON** — centralized record of your entire workflow")
            time.sleep(1)
            st.markdown("**🌐 Dashboard** — interactive viewer to explore results & metadata")
            st.balloons()
    
    with col2:
        st.markdown("### 🧭 Visual Flow Diagram")
    
        svg = """
        <svg width="100%" height="560" xmlns="http://www.w3.org/2000/svg" style="background-color: transparent;">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill="#00d4ff"/>
            </marker>
          </defs>
        
          <!-- VRE Dashed Box -->
          <rect x="35" y="10" width="500" height="120" fill="none" stroke="#00d4ff" stroke-dasharray="5" rx="15"/>
          <text x="250" y="145" fill="#00d4ff" font-size="13">🔁 VRE</text>
        
          <!-- Nodes -->
          <rect x="50" y="20" width="120" height="40" fill="#f94144" rx="10"/>
          <text x="60" y="45" fill="white">📦 DBRepo</text>
        
          <rect x="200" y="20" width="150" height="40" fill="#f3722c" rx="10"/>
          <text x="210" y="45" fill="white">💻 JupyterHub</text>
        
          <rect x="380" y="20" width="130" height="40" fill="#f9c74f" rx="10"/>
          <text x="390" y="45" fill="black">🧠 GitHub</text>
        
          <rect x="200" y="80" width="150" height="40" fill="#90be6d" rx="10"/>
          <text x="210" y="105" fill="white">🗃️ Invenio</text>
        
          <rect x="180" y="180" width="180" height="40" fill="#4d908e" rx="10"/>
          <text x="190" y="205" fill="white">🧰 Metadata Extractor</text>
        
          <rect x="180" y="250" width="180" height="40" fill="#577590" rx="10"/>
          <text x="200" y="275" fill="white">📜 Provenance JSON</text>
        
          <rect x="180" y="320" width="180" height="40" fill="#9b5de5" rx="10"/>
          <text x="200" y="345" fill="white">🌐 Dashboard</text>
        
          <!-- VRE Flow Arrows -->
          <line x1="170" y1="40" x2="200" y2="40" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
          <line x1="350" y1="40" x2="380" y2="40" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
          <line x1="275" y1="60" x2="275" y2="80" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
        
          <!-- Metadata Curved Arrows -->
          <path d="M60 60 C 100 150, 100 150, 190 190" stroke="#00d4ff" fill="none" stroke-width="2" marker-end="url(#arrow)"/>
          <path d="M290 60 C 290 140, 270 140, 270 180" stroke="#00d4ff" fill="none" stroke-width="2" marker-end="url(#arrow)"/>
          <path d="M450 60 C 400 150, 350 150, 360 190" stroke="#00d4ff" fill="none" stroke-width="2" marker-end="url(#arrow)"/>
          <path d="M275 120 C 275 160, 275 160, 275 180" stroke="#00d4ff" fill="none" stroke-width="2" marker-end="url(#arrow)"/>
        
          <!-- Downstream Flow -->
          <line x1="270" y1="220" x2="270" y2="250" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
          <line x1="270" y1="290" x2="270" y2="320" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
        </svg>
        """
        
        components.html(f"""
        <div style="text-align:center; background-color: transparent;">
          {svg}
        </div>
        """, height=580)

elif selected == "📁 Dataset Metadata":
    st.title("📁 Dataset Metadata")
    st.markdown("""
Review comprehensive metadata for the datasets used in your machine learning experiments.

📁 **What you’ll find**:
- Dataset titles, schema info, and repository identifiers
- Source platforms, publication metadata, and DBRepo tags
- Transformation steps: dropped columns, selected features

🔍 **Why it matters**:
- Trace dataset origin and preprocessing stages
- Evaluate FAIR compliance and metadata completeness
""")

    run_ids = df['run_id'].dropna().unique()
    if not run_ids.any():
        st.warning("⚠️ No runs found. Please train a model first.")
    else:
        selected_run = st.selectbox("Select a Run ID", run_ids)
        run_df = df[df["run_id"] == selected_run]

        if run_df.empty:
            st.warning("No metadata available for this run.")
        else:
            flat_row = run_df.iloc[0].to_dict()
            print(flat_row)
            # ==== PROV-O Metadata ====
            st.subheader("🛰️ PROV-O Metadata")
            prov_fields = {
                "Entity": flat_row.get("PROV-O_prov_Entity", "—"),
                "Activity": flat_row.get("PROV-O_prov_Activity", "—"),
                "Agent (Database Creator)": flat_row.get("PROV-O_prov_Agent_database_creator", "—"),
                "Agent (Dataset Creator)": flat_row.get("FAIR_dataset_creator", "—"),
                "Used Source": flat_row.get("PROV-O_prov_used", "—"),
                "Started At": flat_row.get("PROV_startedAtTime", "—"),
                "Ended At": flat_row.get("PROV-O_prov_endedAtTime", "—"),
                "Was Generated By": flat_row.get("PROV-O_prov_wasGeneratedBy", "—"),
                "Was Associated With": flat_row.get("PROV-O_prov_wasAssociatedWith", "—")
            }
            st.dataframe(pd.DataFrame(list(prov_fields.items()), columns=["Field", "Value"]), use_container_width=True)

            # ==== FAIR Dataset Metadata ====
            st.subheader("📚 FAIR Dataset Metadata (e.g., Zenodo,Datacite)")
            fair_fields = {
                "Title": flat_row.get("FAIR_dataset_title", "—"),
                "Creator": flat_row.get("FAIR_dataset_creator", "—"),
                "Publisher": flat_row.get("FAIR_dataset_publisher", "—"),
                "Published": flat_row.get("FAIR_dataset_publication_date", "—"),
                "DOI": flat_row.get("FAIR_dataset_identifier", "—"),
                "Description": flat_row.get("FAIR_dataset_description", "—"),
                "Documentation": flat_row.get("FAIR_dataset_documentation", "—"),
                "Access URL": flat_row.get("FAIR_dataset_access_url", "—")
            }
            st.dataframe(pd.DataFrame(list(fair_fields.items()), columns=["Field", "Value"]), use_container_width=True)

            # ==== Internal DBRepo Metadata ====
            st.subheader("🏛️ Internal DBRepo Metadata")
            db_fields = {
                "Title": flat_row.get("Internal_DBRepo_database_title", "—"),
                "Creator": flat_row.get("Internal_DBRepo_database_creator", "—"),
                "Publisher": flat_row.get("Internal_DBRepo_database_publisher", "—"),
                "Access URL": flat_row.get("Internal_DBRepo_data_source", "—"),
                "Raw Columns": flat_row.get("param_Internal_DBRepo_columns_raw", "—"),
                "Dropped Columns": flat_row.get("param_Internal_DBRepo_dropped_columns", "—"),
                "Final Features": flat_row.get("param_Internal_DBRepo_feature_names", "—"),
                "Records": flat_row.get("param_Internal_DBRepo_n_records", "—"),
                "Last Modified": flat_row.get("Internal_DBRepo_table_last_modified", "—")
            }
            st.dataframe(pd.DataFrame(list(db_fields.items()), columns=["Field", "Value"]), use_container_width=True)

            # ==== Preprocessing ====
            st.subheader("🧪 Preprocessing & Transformations")
            preprocessing_raw = flat_row.get("MLSEA_dataPreprocessing", None)
            if preprocessing_raw:
                try:
                    preprocessing = json.loads(preprocessing_raw)
                    st.json(preprocessing)
                except Exception as e:
                    st.warning(f"⚠️ Could not parse preprocessing JSON: {e}")
            else:
                st.warning("No preprocessing trace captured.")

elif selected == "🧠 ML Model Metadata":
    st.title("🧠 ML Model Metadata")
    st.markdown("""
Explore structured ML model metadata from each experiment.

🔍 What’s covered:
- Hyperparameters, metrics, and justifications  
- Compute environment and training timeline  
- FAIR4ML and MLSEA metadata structures  
""")

    run_ids = df['run_id'].dropna().unique()
    if not run_ids.any():
        st.warning("⚠️ No runs found. Please train a model first.")
    else:
        selected_run = st.selectbox("Select a Run ID", run_ids)
        row = df[df["run_id"] == selected_run].iloc[0].to_dict()

        def clean_val(v):
            if isinstance(v, (dict, list)):
                return json.dumps(v, indent=2)
            elif v is None:
                return "—"
            return str(v)

        def section(title, fields: dict):
            st.subheader(title)
            cleaned = {k: clean_val(v) for k, v in fields.items()}
            display_df = pd.DataFrame(list(cleaned.items()), columns=["Field", "Value"])
            st.dataframe(display_df, use_container_width=True)

        # 🚀 Overview
        section("🚀 Model Overview", {
            "Model Name": row.get("ML_EXP_model_name"),
            "Model Architecture": row.get("MLSEA_modelArchitecture"),
            "Notebook": row.get("ML_EXP_notebook_name"),
            "Run Name": row.get("mlflow.runName"),
            "Experiment ID": row.get("MLSEA_experimentId")
        })

        # 🧠 Hyperparameters
        try:
            hyper = json.loads(row.get("MLSEA_hyperparameters", "{}"))
        except Exception:
            hyper = {}
        section("🧠 Model Hyperparameters", hyper)

        # 📊 Evaluation Metrics
        try:
            metrics = json.loads(row.get("MLSEA_evaluationMetrics", "{}"))
        except Exception:
            metrics = {}
        section("📊 Evaluation Metrics", metrics)

        # 🧰 Compute Environment
        try:
            env = json.loads(row.get("MLSEA_computeEnvironment", "{}"))
        except Exception:
            env = {}
        section("🧰 Compute Environment", env)

        # 🧪 Training Metadata
        section("🧪 Training Timeline", {
            "Training Start Time": row.get("ML_EXP_training_start_time"),
            "Training End Time": row.get("ML_EXP_training_end_time"),
            "Training Procedure": row.get("MLSEA_trainingProcedure"),
            "Previous Model": row.get("MLSEA_previousModelRunId"),
            "Model Path": row.get("MLSEA_modelPath")
        })

        # 📋 Justifications
        justifications = {
            k.replace("MLSEA_justification_", ""): v
            for k, v in row.items()
            if k.startswith("MLSEA_justification_")
        }
        if justifications:
            section("📋 Configuration Justifications", justifications)
        else:
            st.info("ℹ️ No justifications recorded for this run.")

        # 📌 Additional Insights
        section("📌 Additional Insights", {
            "Performance Notes": row.get("MLSEA_performanceInterpretation"),
            "Preprocessing Hash": row.get("MLSEA_preprocessing_hash"),
            "Training Code Snapshot": row.get("MLSEA_trainingCodeSnapshot")
        })

elif selected == "📊 Model Plots":
    st.title("📊 Model Explainability & Evaluation Plots")
    st.markdown("""
Visualize how your machine learning model is performing — and understand **why** it's making the predictions it does.

🔗 This section links each plot back to the run ID, dataset, and model used to generate it.
""")

    plot_folders = glob.glob(os.path.join("ML_EXP_plots", "RandomForest_Iris_v*"))
    plot_folders = [os.path.basename(folder) for folder in plot_folders]

    if not plot_folders:
        st.warning("⚠️ No plot folders found.")
    else:
        # Map model filename (without .pkl) to run metadata
        model_to_run = {}
        for _, row in df.iterrows():
            model_path = row.get("MLSEA_modelPath", "")
            if model_path:
                model_key = os.path.splitext(os.path.basename(model_path))[0]
                model_to_run[model_key] = row

        selected_folder = st.selectbox("Select a Run (for Plots)", plot_folders)

        run_data = model_to_run.get(selected_folder)
        if run_data is not None:
            st.success(f"✅ Matched Run ID: `{run_data.get('run_id', '—')}`")

            with st.expander("📋 Extended Metadata"):

                def safe_str(val):
                    if isinstance(val, (dict, list)):
                        return json.dumps(val)
                    elif val is None:
                        return "—"
                    return str(val)

                meta_preview = {
                    "Run ID": run_data.get("run_id", "—"),
                    "Model Name": run_data.get("ML_EXP_model_name", "—"),
                    "Dataset Title": run_data.get("FAIR_dataset_title", "—"),
                    "Training Start": run_data.get("ML_EXP_training_start_time", "—"),
                    "Training End": run_data.get("ML_EXP_training_end_time", "—"),
                    "Accuracy (Test)": run_data.get("metric_ML_EXP_accuracy", "—"),
                    "F1 Macro (Test)": run_data.get("metric_ML_EXP_f1_macro", "—"),
                    "Precision (Test)": run_data.get("metric_ML_EXP_precision_macro", "—"),
                    "Recall (Test)": run_data.get("metric_ML_EXP_recall_macro", "—"),
                    "ROC AUC (Test)": run_data.get("metric_ML_EXP_roc_auc", "—"),
                    "Training Accuracy": run_data.get("metric_training_accuracy_score", "—"),
                    "Target Variable": run_data.get("FAIR4ML_target_variable", "—"),
                    "Serialization Format": run_data.get("FAIR4ML_serializationFormat", "—"),
                    "Model Path": run_data.get("MLSEA_modelPath", "—"),
                    "Improved From": run_data.get("MLSEA_improvedFrom", "—"),
                    "Training Code Snapshot": run_data.get("MLSEA_trainingCodeSnapshot", "—"),
                    "Training Procedure": run_data.get("MLSEA_trainingProcedure", "—"),
                }

                # Add selected hyperparameters
                try:
                    hparams = json.loads(run_data.get("MLSEA_hyperparameters", "{}"))
                except Exception:
                    hparams = {}
                for key, val in hparams.items():
                    meta_preview[f"Hyperparam → {key}"] = val

                # Add preprocessing summary
                try:
                    prep = json.loads(run_data.get("MLSEA_dataPreprocessing", "{}"))
                except Exception:
                    prep = {}
                for key in ["dropped_columns", "final_feature_columns", "target_column"]:
                    if key in prep:
                        meta_preview[f"Preprocessing → {key}"] = prep[key]

                # Clean all values before rendering
                cleaned_meta_preview = {k: safe_str(v) for k, v in meta_preview.items()}
                meta_df = pd.DataFrame(list(cleaned_meta_preview.items()), columns=["Field", "Value"])
                st.dataframe(meta_df, use_container_width=True)

        else:
            st.warning("⚠️ No run metadata found for this plot folder.")

        plot_options = {
            "Feature Importances": "feature_importances.png",
            "Confusion Matrix": "confusion_matrix.png",
            "SHAP Summary": "shap_summary.png",
            "ROC Curve (Class 0)": "roc_curve_cls_0.png",
            "Precision-Recall (Class 0)": "pr_curve_cls_0.png"
        }

        selected_plot = st.selectbox("Choose Plot", list(plot_options.keys()))
        plot_path = os.path.join("ML_EXP_plots", selected_folder, plot_options[selected_plot])

        if os.path.exists(plot_path):
            plot_width = st.slider("Adjust Plot Width", 400, 1000, 600)
            st.image(plot_path, caption=f"{selected_plot} — {selected_folder}", width=plot_width)

            # Explain each plot
            explanations = {
                "Feature Importances": "Shows which features contribute most to predictions.",
                "SHAP Summary": "SHAP values show feature impact and distribution.",
                "ROC Curve": "Visualizes true vs. false positive rates.",
                "Precision-Recall": "Helps evaluate classifier performance under class imbalance.",
                "Confusion Matrix": "Compares predicted vs. actual outcomes."
            }
            for name, desc in explanations.items():
                if name.split()[0] in selected_plot:
                    st.markdown(f"**Interpretation:** {desc}")
        else:
            st.error("❌ Selected plot file not found.")


elif selected == "🛰️ Provenance Trace":
    st.title("🛰️ Provenance Trace")
    st.markdown("""
Use this view to inspect detailed provenance metadata for a specific training run — and optionally compare it with another.

📌 **Use Case Highlights**:
- **Provenance & Reproducibility**: Trace how a model result was produced — including versions, parameters, and preprocessing.
- **Training Configuration & Evaluation**: Compare hyperparameters, strategies, and performance outcomes across runs.
    """)

    run_ids = df['run_id'].dropna().unique()
    selected_run = st.selectbox("Select Run 1", run_ids)
    run_data_1 = df[df['run_id'] == selected_run].iloc[0].to_dict()

    compare_mode = st.checkbox("🔁 Compare with another run")
    run_data_2 = None
    second_run = None

    if compare_mode:
        second_run = st.selectbox("Select Run 2", [r for r in run_ids if r != selected_run])
        run_data_2_df = df[df['run_id'] == second_run]
        if not run_data_2_df.empty:
            run_data_2 = run_data_2_df.iloc[0].to_dict()

    def get_provenance_fields(run_data):
        return {
            "Run ID": run_data.get("run_id", "—"),
            "Dataset Title": run_data.get("FAIR_dataset_title", "—"),
            "Dataset Version": run_data.get("ML_EXP_dataset_version", "—"),
            "Dataset Source URL": run_data.get("FAIR_dataset_access_url", "—"),
            "Notebook Name": run_data.get("ML_EXP_notebook_name", "—"),
            "Model Path": run_data.get("MLSEA_modelPath", "—"),
            "Model Architecture": run_data.get("MLSEA_modelArchitecture", "—"),
            "Training Code Snapshot": run_data.get("MLSEA_trainingCodeSnapshot", "—"),
            "GIT Commit Hash": run_data.get("GIT_current_commit_hash", "—"),
            "GIT Commit URL": run_data.get("GIT_current_commit_url", "—"),
            "Preprocessing Hash": run_data.get("MLSEA_preprocessing_hash", "—"),
            "Preprocessing Timestamp": json.loads(run_data.get("MLSEA_dataPreprocessing", "{}")).get("preprocessing_timestamp", "—"),
            "Training Start Time": run_data.get("ML_EXP_training_start_time", "—"),
            "Training End Time": run_data.get("ML_EXP_training_end_time", "—"),
            "Database Title": run_data.get("Internal_DBRepo_database_title", "—"),
            "Database Creator": run_data.get("Internal_DBRepo_database_creator", "—"),
            "Database Last Modified": run_data.get("Internal_DBRepo_table_last_modified", "—"),
            "Generated By": run_data.get("PROV-O_prov_wasGeneratedBy", "—"),
            "Used Source": run_data.get("PROV-O_prov_used", "—"),
            "Activity": run_data.get("PROV-O_prov_Activity", "—")
        }

    def get_config_and_eval_fields(run_data):
        return {
            "Target Variable": run_data.get("FAIR4ML_target_variable", "—"),
            "Split Strategy": run_data.get("MLSEA_trainingProcedure", "—"),
            "Model Architecture": run_data.get("MLSEA_modelArchitecture", "—"),
            "Accuracy (Test)": run_data.get("metric_ML_EXP_accuracy", "—"),
            "F1 Score (Test)": run_data.get("metric_ML_EXP_f1_macro", "—"),
            "Precision (Test)": run_data.get("metric_ML_EXP_precision_macro", "—"),
            "Recall (Test)": run_data.get("metric_ML_EXP_recall_macro", "—"),
            "ROC AUC (Test)": run_data.get("metric_ML_EXP_roc_auc", "—"),
            "Accuracy (Train)": run_data.get("metric_training_accuracy_score", "—"),
            "F1 Score (Train)": run_data.get("metric_training_f1_score", "—"),
            "Precision (Train)": run_data.get("metric_training_precision_score", "—"),
            "Recall (Train)": run_data.get("metric_training_recall_score", "—"),
            "Loss (Train)": run_data.get("metric_training_log_loss", "—"),
            "Hyperparam → n_estimators": run_data.get("param_n_estimators", "—"),
            "Hyperparam → max_depth": run_data.get("param_max_depth", "—"),
            "Hyperparam → min_samples_split": run_data.get("param_min_samples_split", "—"),
            "Hyperparam → min_samples_leaf": run_data.get("param_min_samples_leaf", "—"),
            "Hyperparam → criterion": run_data.get("param_criterion", "—"),
            "Hyperparam → max_features": run_data.get("param_max_features", "—"),
            "Hyperparam → bootstrap": run_data.get("param_bootstrap", "—"),
            "Hyperparam → oob_score": run_data.get("param_oob_score", "—"),
            "Hyperparam → class_weight": run_data.get("param_class_weight", "—"),
            "Hyperparam → random_state": run_data.get("param_random_state", "—")
        }

    def display_comparison(title, data_fn):
        st.subheader(title)
        data1 = data_fn(run_data_1)
    
        if compare_mode and run_data_2:
            data2 = data_fn(run_data_2)
    
            # Force all content to string to prevent Arrow type errors
            df_display = pd.DataFrame({
                "Field": [str(k) for k in data1.keys()],
                f"Run 1 ({selected_run})": [str(data1.get(k, "—")) for k in data1.keys()],
                f"Run 2 ({second_run})": [str(data2.get(k, "—")) for k in data1.keys()]
            })
    
            def highlight_diff(row):
                styles = []
                for i, cell in enumerate(row):
                    if i == 0:
                        styles.append("")  # Field column
                    elif row[1] != row[2] and i in [1, 2]:
                        styles.append("background-color: #fbe8e8; color: black")
                    else:
                        styles.append("")
                return styles
    
            st.dataframe(df_display.style.apply(highlight_diff, axis=1), use_container_width=True)
    
        else:
            df_display = pd.DataFrame({
                "Field": [str(k) for k in data1.keys()],
                "Value": [str(v) for v in data1.values()]
            })
            st.dataframe(df_display, use_container_width=True)


    # Render both comparison tables
    display_comparison("🔍 Provenance & Reproducibility Details", get_provenance_fields)
    display_comparison("🧪 Configuration & Evaluation Strategy", get_config_and_eval_fields)

elif selected == "🧭 Model-Dataset Mapping":
    st.title("🧭 Model-Dataset Mapping")
    st.markdown("""
Gain insights into which machine learning models were trained on which datasets — and how they performed.

🔗 **This view helps answer:**
- Which ML models were trained on which datasets?
- What dataset versions were used?
- What were the training outcomes?

📌 **Details shown:**
- Model name & architecture
- Dataset title, version, and access URL
- Accuracy, F1 score, ROC AUC (test set)
""")

    try:
        # Build a structured table from the main df
        mapping_records = []
        for _, row in df.iterrows():
            model_name = row.get("ML_EXP_model_name", "—")
            model_arch = row.get("MLSEA_modelArchitecture", "—")
            dataset_title = row.get("FAIR_dataset_title", "—")
            dataset_version = row.get("ML_EXP_dataset_version", "—")
            dataset_url = row.get("FAIR_dataset_access_url", "—")
            accuracy = row.get("metric_ML_EXP_accuracy", "—")
            f1_score = row.get("metric_ML_EXP_f1_macro", "—")
            roc_auc = row.get("metric_ML_EXP_roc_auc", "—")
            run_id = row.get("run_id", "—")

            mapping_records.append({
                "Run ID": run_id,
                "Model Name": model_name,
                "Architecture": model_arch,
                "Dataset Title": dataset_title,
                "Dataset Version": dataset_version,
                "Dataset Access URL": dataset_url,
                "Accuracy (Test)": accuracy,
                "F1 Score (Test)": f1_score,
                "ROC AUC (Test)": roc_auc
            })

        if mapping_records:
            df_mapping = pd.DataFrame(mapping_records)
            st.dataframe(df_mapping, use_container_width=True)
        else:
            st.warning("⚠️ No valid model-dataset mappings found.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")


elif selected == "⚠️ Deprecated Code Check":
    st.title("⚠️ Deprecated Code Check")
    st.markdown("""
    Identify ML experiment runs that were executed using outdated or deprecated code versions.

    🔍 **How it works**:  
    Provide one or more Git commit hashes below (e.g., from GitHub history). The system will compare these against the commit hashes recorded in your experiment metadata and flag any matching runs.

    🧪 **Use cases**:
    - Track experiments run on stale forks or branches  
    - Maintain codebase hygiene across collaborators  
    - Ensure reproducibility by auditing legacy runs

    💡 You can enter multiple commit hashes (one per line).
    """)
    
        # Input for deprecated commits
    deprecated_commits_input = st.text_area(
            "Enter deprecated commit hashes (one per line):",
            height=100
        )
    
    if deprecated_commits_input:
            deprecated_commits = [line.strip() for line in deprecated_commits_input.strip().split('\n') if line.strip()]
            if st.button("Run Check"):
                try:
                    results = detect_deprecated_code(df, deprecated_commits=deprecated_commits)
                    if results:
                        st.write("### Runs using deprecated commits:")
                        st.dataframe(pd.DataFrame(results))
                    else:
                        st.success("No runs found with deprecated commits.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
            st.info("Please enter at least one deprecated commit hash.")


elif selected == "📣 Notify Outdated Forks":
    st.title("📣 Notify Outdated Forks")
    st.markdown("""
Detect whether collaborators' forks of your GitHub repository are out-of-date with the main branch — and automatically notify them by opening a GitHub Issue.

📣 **What it does**:
- Checks the latest commit in the main repository
- Compares it with the latest commit in each fork
- Identifies forks that are behind
- Sends a polite issue notification to encourage syncing

🔧 **How to use**:
1. Enter the GitHub **owner**, **repository name**, and your **personal access token**  
2. Click **🔔 Notify Fork Owners**  
3. A GitHub Issue will be opened for any forks that are not up-to-date

💡 Useful for collaborative research, codebase alignment, and project maintenance.
""")
    # Input fields
    owner = st.text_input("GitHub Owner", value="reema-dass26")
    repo = st.text_input("Repository Name", value="REPO")
    token = st.text_input("GitHub Token", type="password")

    if st.button("🔔 Notify Fork Owners"):
        if not all([owner, repo, token]):
            st.warning("Please fill in all fields before proceeding.")
        else:
            with st.spinner("Checking forks..."):
                try:
                    headers = {
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }

                    main_commits = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}/commits",
                        headers=headers,
                        params={"per_page": 1}
                    )
                    main_commits.raise_for_status()
                    new_commit_hash = main_commits.json()[0]["sha"]
                    st.success(f"✅ Latest commit: `{new_commit_hash}`")

                    forks_resp = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}/forks",
                        headers=headers
                    )
                    forks_resp.raise_for_status()
                    forks = forks_resp.json()

                    outdated = []
                    for fork in forks:
                        fork_owner = fork["owner"]["login"]
                        fork_comm = requests.get(
                            fork["url"] + "/commits",
                            headers=headers,
                            params={"per_page": 1}
                        )
                        if fork_comm.status_code != 200:
                            st.warning(f"⚠️ Could not check @{fork_owner}")
                            continue

                        fork_sha = fork_comm.json()[0]["sha"]
                        if fork_sha != new_commit_hash:
                            outdated.append(fork_owner)

                    if outdated:
                        st.warning(f"These forks are outdated: {', '.join(outdated)}")

                        title = "🔔 Notification: Your fork is behind the latest commit"
                        body = (
                            f"Hi {' '.join(f'@{u}' for u in outdated)},\n\n"
                            f"The main repository has been updated to commit `{new_commit_hash}`.\n"
                            "Please consider pulling the latest changes to stay in sync.\n\n"
                            "Thanks!"
                        )

                        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
                        issue_resp = requests.post(
                            issue_url,
                            headers=headers,
                            json={"title": title, "body": body}
                        )

                        if issue_resp.status_code == 201:
                            issue_link = issue_resp.json().get("html_url")
                            st.success(f"Issue created: [View Issue]({issue_link})")
                        else:
                            st.error("❌ Failed to create issue.")
                            st.code(issue_resp.text)
                    else:
                        st.success("✅ All forks are up-to-date!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif selected == "📘 Researcher Justifications":
    st.title("📘 Researcher Justifications")
    st.markdown("""
This section displays all recorded **justifications** provided by the researcher 
for specific modeling decisions, such as hyperparameter choices, dataset version, and evaluation criteria.

🧠 These justifications help ensure:
- **Transparency** in decision-making  
- **Explainability** of configuration  
- **Reproducibility** of results  
""")

    # 1. Detect provenance folders
    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]

    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        # 2. Select a folder
        selected_folder = st.selectbox("Select a Run (for Justifications)", provenance_folders)

        # 3. Construct path to summary JSON
        summary_path = os.path.join(
            "MODEL_PROVENANCE",
            selected_folder,
            f"{selected_folder}_run_summary.json"
        )

        # 4. Load and display justifications
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            justifications = {
                k.replace("justification_", "").replace("_", " ").capitalize(): v
                for k, v in summary.get("ML_EXP_tags", {}).items()
                if k.startswith("justification_") and isinstance(v, str) and v.strip()
            }

            if justifications:
                df_just = pd.DataFrame(
                    list(justifications.items()),
                    columns=["Modeling Decision", "Justification"]
                )
                st.success(f"✅ Loaded justifications for `{selected_folder}`")
                st.write("### 📋 Researcher Justification Table")
                st.dataframe(df_just, use_container_width=True)
            else:
                st.info("ℹ️ No justifications were provided in this run.")

        except Exception as e:
            st.error(f"❌ Failed to load justification data: {e}")

elif selected == "📚 Invenio Metadata":
    st.title("📚 Invenio Metadata")
    st.markdown("""
    View metadata fetched from INVENIO, after data being published).
    Here you can see the details of the lastest model and other files 
    🔍 This includes:
    - Title, creators, publication date
    - PID and status info
    - Files and stats (views/downloads)
    """)
    # 1. List all available MODEL_PROVENANCE folders
    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]
    
    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        # 2. User selects which run
        selected_run = st.selectbox("Select a Run for Invenio Metadata", provenance_folders)
    
        # 3. Build the path to the summary JSON
        summary_path = os.path.join("MODEL_PROVENANCE", selected_run, f"{selected_run}_run_summary.json")
    
        try:
            with open(summary_path, "r") as f:
                run_data = json.load(f)
            invenio_meta = run_data.get("invenio_metadata", {})
    
            if invenio_meta:
                # 4. Neat DataFrame view for quick glance
                df_view = pd.DataFrame([{
                    "Title": invenio_meta.get("title", ""),
                    "Creator": invenio_meta.get("creator", ""),
                    "Published": invenio_meta.get("publication_date", ""),
                    "Status": invenio_meta.get("status", ""),
                    "Views": invenio_meta.get("views", 0),
                    "Downloads": invenio_meta.get("downloads", 0)
                }])
    
                st.header("📚 Invenio Metadata Overview")
                st.dataframe(df_view, use_container_width=True)
    
                # 5. Show full file info
                st.header("📁 Files in Publication")
                files_list = invenio_meta.get("files", [])
                if files_list:
                    st.json(files_list)
                else:
                    st.info("ℹ️ No files recorded in the publication.")
            else:
                st.warning("ℹ️ No `invenio_metadata` found for this run.")
    
        except Exception as e:
            st.error(f"❌ Error loading Invenio metadata: {e}")

elif selected == "📤 Export Provenance":
    st.title("📤 Export Provenance")

    # 1. List available runs
    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]

    if not provenance_folders:
        st.warning("⚠️ No provenance data available.")
    else:
        # 2. Run selection
        selected_run = st.selectbox("Select a Run ID", provenance_folders)

        # 3. Format selection
        export_format = st.radio(
            "Choose Export Format",
            options=["JSON", "JSON-LD", "RDF/XML"]
        )

        # 4. Base folder
        base_path = os.path.join("MODEL_PROVENANCE", selected_run)

        # 5. File discovery
        file_path = None
        viz_path = None

        if export_format == "JSON":
            file_path = os.path.join(base_path, f"{selected_run}_run_summary.json")
        elif export_format == "JSON-LD":
            jsonld_files = glob.glob(os.path.join(base_path, "*.jsonld"))
            file_path = jsonld_files[0] if jsonld_files else None
            viz_candidates = glob.glob(os.path.join(base_path, "*JSONLD_viz.png"))
            viz_path = viz_candidates[0] if viz_candidates else None
        elif export_format == "RDF/XML":
            xml_files = glob.glob(os.path.join(base_path, "*.xml"))
            file_path = xml_files[0] if xml_files else None
            viz_candidates = glob.glob(os.path.join(base_path, "*RDFXML_viz.png"))
            viz_path = viz_candidates[0] if viz_candidates else None

        # 6. Download and Preview
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as file:
                file_content = file.read()

            st.download_button(
                label=f"📥 Download {export_format}",
                data=file_content,
                file_name=os.path.basename(file_path),
                mime="application/json" if "json" in file_path.lower() else "application/xml"
            )

            # 7. If Visualization exists
            if viz_path and os.path.exists(viz_path):
                st.image(viz_path, caption=f"Visualization for {export_format}", use_column_width=True)
            else:
                st.info("🔍 No visualization available for this format.")

        else:
            st.error(f"❌ {export_format} file not found for this run.")

            
elif selected == "🧨 Error & Version Impact":
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    st.title("🧨 Error & Version Impact Analysis")
    st.markdown("""
Detect which ML experiments were affected by **outdated code versions**.

🔍 **Why it matters**:
- Identifies affected researchers  
- Flags experiments needing retraining  
- Supports reproducibility  
    """)

    def get_current_git_commit():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except Exception:
            return None

    current_hash = get_current_git_commit()
    version_map = {}

    if 'GIT_current_commit_hash' in df.columns:
        unique_commits = df['GIT_current_commit_hash'].dropna().unique()
        st.markdown("### 🏷️ Tag Git Commits with Version Labels")
        for commit in unique_commits:
            default_tag = "untagged"
            if 'GIT_code_version' in df.columns:
                tags = df[df['GIT_current_commit_hash'] == commit]['GIT_code_version'].dropna().unique()
                if len(tags) > 0:
                    default_tag = tags[0]
            tag = st.text_input(f"Version tag for `{commit[:8]}...`", value=default_tag)
            version_map[commit] = tag or "untagged"

    if current_hash:
        st.markdown(f"### 📌 Current Git Commit: `{current_hash}`")

    if 'GIT_current_commit_hash' in df.columns:
        df['GIT_code_version'] = df['GIT_current_commit_hash'].map(version_map).fillna("untagged")
        st.markdown("### 🧾 Existing Commit Tags:")
        st.dataframe(df[['run_id', 'GIT_current_commit_hash', 'GIT_code_version']], use_container_width=True)

    deprecated_versions_input = st.text_area("Enter deprecated version tags (one per line):", height=100)
    simulate_current = st.checkbox("☢️ Also mark current local commit as deprecated")
    current_version_tag = version_map.get(current_hash, "untagged")
    if simulate_current:
        deprecated_versions_input += f"\n{current_version_tag}"
        st.info(f"☢️ Added current version `{current_version_tag}` to deprecated list.")

    deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]

    def detect_deprecated_versions(df, deprecated_versions):
        if 'GIT_code_version' not in df.columns:
            return pd.DataFrame()
        affected = df[df['GIT_code_version'].isin(deprecated_versions)].copy()
        if 'GIT_user_email' in df.columns:
            affected['github_user'] = df['GIT_user_email'].str.extract(r"\+([a-zA-Z0-9\-]+)@users")[0]
        return affected

    results_df = pd.DataFrame()
    if st.button("🚨 Detect Impacted Runs"):
        if not deprecated_versions:
            st.warning("Please enter at least one deprecated version.")
        else:
            detected = detect_deprecated_versions(df, deprecated_versions)
            if detected.empty:
                st.success("✅ No impacted runs found.")
            else:
                st.warning("⚠️ Impacted Experiments Detected:")
                st.session_state.results_df = detected
                st.dataframe(st.session_state.results_df, use_container_width=True)


    if not st.session_state.results_df.empty:
        st.markdown("### 📣 Notify Affected Users via GitHub")
        with st.expander("🔐 GitHub Authentication"):
            owner = st.text_input("GitHub Owner", value="reema-dass26")
            repo = st.text_input("Repository Name", value="REPO")
            token = st.text_input("GitHub Token", type="password")
    
            if st.button("📬 Notify Affected Users"):
                if not all([owner, repo, token]):
                    st.warning("❗ Provide all GitHub credentials.")
                else:
                    try:
                        impacted_users = st.session_state.results_df['github_user'].dropna().unique()
                        user_tags = " ".join(f"@{u}" for u in impacted_users)
                        issue_body = (
                            f"The following experiments were run on deprecated versions:\n\n"
                            f"- Versions: {', '.join(set(deprecated_versions)) or 'N/A'}\n\n"
                            f"{user_tags}\n\n"
                            "Please retrain or validate your experiments.\n\n"
                            "— Provenance Dashboard"
                        )
    
                        headers = {
                            "Authorization": f"token {token}",
                            "Accept": "application/vnd.github+json"
                        }
                        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
                        resp = requests.post(issue_url, headers=headers, json={"title": "🚨 Deprecated Version Used", "body": issue_body})
    
                        if resp.status_code == 201:
                            st.success(f"✅ GitHub Issue Created: [View Issue]({resp.json().get('html_url')})")
                        else:
                            st.error(f"❌ GitHub Issue failed: {resp.status_code}")
                            st.code(resp.text)
                    except Exception as e:
                        st.error(f"Exception occurred: {str(e)}")

