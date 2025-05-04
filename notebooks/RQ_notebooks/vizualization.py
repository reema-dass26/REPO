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
    commit_col = 'tag_git_current_commit_hash'
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
                summary = json.loads(content)  # <- Load content, not file handle!

            row = {"run_id": summary.get("run_id", "")}
            row.update({f"param_{k}": v for k, v in summary.get("params", {}).items()})
            row.update({f"metric_{k}": v for k, v in summary.get("metrics", {}).items()})
            row.update({f"tag_{k}": v for k, v in summary.get("tags", {}).items()})
            rows.append(row)

        except Exception as e:
            st.error(f"❌ Error loading file {file_path}: {e}")

    if not rows:
        st.warning("⚠️ No valid run summary data could be loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    st.success(f"✅ Loaded {len(df)} runs with {len(df.columns)} columns.")
   
    
    return df

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
    with open(path, "r") as f:
        js = json.load(f)

    justifications = {
        k: v for k, v in js.get("tags", {}).items()
        if k.startswith("justification_")
    }
    rows = [
        {"Decision": k.replace("justification_", ""), "Justification": v}
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
            "⚠️ Deprecated Code Check",
            "🧭 Model-Dataset Mapping",
            "📣 Notify Outdated Forks",
            "📘 Researcher Justifications",
            "📚 Invenio Metadata",
            "📤 Export Provenance",
            "🧨 Error & Version Impact"

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


# Main content area
if selected == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.markdown("## 👋 Welcome to the End to End Provenance Dashboard")
    st.markdown("""
This dashboard is designed to assist researchers and practitioners in managing and understanding the provenance of machine learning experiments conducted in virtual research environments (VREs). It provides an interactive and structured overview of key aspects of ML workflows, enabling **traceability, reproducibility, and transparency**.

### 🧩 Key Features:

- 🧬 **Dataset Metadata**  
  Inspect detailed metadata of datasets used in experiments, including authorship, publication, versioning, source platform, and database integration.

- 🧠 **ML Training Configuration**  
  Explore the complete training setup including model hyperparameters, platform environment (e.g., Python, NumPy, scikit-learn versions), sampling details, and system configurations.

- 📊 **Model Plots**  
  Visualize model performance and interpretation outputs such as:
  - Feature importance
  - Confusion matrix
  - ROC & Precision-Recall curves
  - SHAP explanation plots

- 🛰️ **Provenance Trace**  
  Analyze the flow and transformation of data through various stages of your ML pipeline. Includes utilities to:
  - Trace preprocessing steps
  - Evaluate feature importance by drop-impact analysis
  - Filter runs by configuration and accuracy thresholds

- ⚠️ **Deprecated Code Check**  
  Identify and flag experiments that rely on outdated or deprecated source code versions (e.g., old Git commits), ensuring codebase hygiene and reproducibility.

- 🧭 **Model-Dataset Mapping**  
  Get a clear view of which models were trained on which datasets. See linked metadata like DOI, publication date, and publisher, helping maintain proper attribution and context.

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
elif selected == "🧨 Error & Version Impact":
    st.title("🧨 Error & Version Impact Analysis")
    st.markdown("""
    Detect which ML experiments were affected by **outdated code versions** or **deprecated dataset versions**.

    🔍 **Why it matters**:
    - Identifies affected researchers or notebooks
    - Flags experiments that may need retraining
    - Supports version hygiene and reproducibility
    """)

    st.markdown("### 🧾 Existing Commits & Versions in Dataset:")
    if 'tag_git_current_commit_hash' in df.columns:
        st.code("\n".join(df['tag_git_current_commit_hash'].dropna().unique().tolist()), language='text')
    if 'tag_dataset_version' in df.columns:
        st.code("\n".join(df['tag_dataset_version'].dropna().unique().tolist()), language='text')

    # Input areas for deprecated resources
    deprecated_commits_input = st.text_area("Enter deprecated commit hashes (one per line):", height=100)
    deprecated_versions_input = st.text_area("Enter deprecated dataset versions (one per line):", height=100)

    # Process input
    deprecated_commits = [c.strip() for c in deprecated_commits_input.splitlines() if c.strip()]
    deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]

    def detect_deprecated_resources(
        df: pd.DataFrame,
        deprecated_commits: List[str],
        deprecated_dataset_versions: List[str]
    ) -> pd.DataFrame:
        commit_col = 'tag_git_current_commit_hash'
        version_col = 'tag_dataset_version'
        candidate_authors = ['tag_executed_by', 'tag_user', 'tag_notebook_name', 'param_author']
        author_col = next((col for col in candidate_authors if col in df.columns), None)

        mask_commit = df[commit_col].isin(deprecated_commits) if commit_col in df.columns else False
        mask_version = df[version_col].isin(deprecated_dataset_versions) if version_col in df.columns else False
        impacted = df[mask_commit | mask_version]

        cols = ['run_id', commit_col, version_col, 'tag_mlflow.runName']
        if author_col:
            cols.append(author_col)

        return impacted[cols]

    if st.button("🚨 Detect Impacted Runs"):
        if not deprecated_commits and not deprecated_versions:
            st.warning("Please enter at least one deprecated commit or dataset version.")
        else:
            results_df = detect_deprecated_resources(df, deprecated_commits, deprecated_versions)
            if results_df.empty:
                st.success("✅ No impacted runs found.")
            else:
                st.warning("⚠️ Impacted Experiments Detected:")
                st.dataframe(results_df, use_container_width=True)

                # Group by user if available
                candidate_authors = ['tag_executed_by', 'tag_user', 'tag_notebook_name', 'param_author']
                found_author_col = next((col for col in candidate_authors if col in results_df.columns), None)

                if found_author_col:
                    summary = results_df[found_author_col].value_counts().reset_index()
                    summary.columns = ['User', 'Impacted Runs']
                    st.markdown("### 👥 Affected Users:")
                    st.dataframe(summary, use_container_width=True)
                    # GitHub notification section
                st.markdown("---")
                st.markdown("### 📣 Notify via GitHub Issue")

                with st.expander("🔐 GitHub Authentication"):
                    github_owner = st.text_input("GitHub Owner (username or org)", key="gh_owner")
                    github_repo = st.text_input("Repository Name", key="gh_repo")
                    github_token = st.text_input("GitHub Personal Access Token", type="password", key="gh_token")

                if st.button("📬 Notify via GitHub"):
                    if not all([github_owner, github_repo, github_token]):
                        st.warning("❗ Please provide GitHub credentials above.")
                    else:
                        try:
                            impacted_users = results_df[found_author_col].dropna().unique().tolist()
                            user_tags = " ".join(f"@{u}" for u in impacted_users if re.match(r"^[a-zA-Z0-9\-_]+$", u))

                            issue_title = "🚨 Deprecated Resources Used in ML Experiments"
                            issue_body = (
                                f"Dear team,\n\n"
                                f"The following experiments were found using **deprecated code or dataset versions**:\n\n"
                                f"- **Commits**: {', '.join(deprecated_commits) if deprecated_commits else 'N/A'}\n"
                                f"- **Dataset Versions**: {', '.join(deprecated_versions) if deprecated_versions else 'N/A'}\n\n"
                                f"{user_tags}\n\n"
                                f"Please consider retraining or validating your runs.\n\n"
                                f"— Automated notifier from the Provenance Dashboard"
                            )

                            headers = {
                                "Authorization": f"token {github_token}",
                                "Accept": "application/vnd.github+json"
                            }
                            issue_api_url = f"https://api.github.com/repos/{github_owner}/{github_repo}/issues"
                            payload = {"title": issue_title, "body": issue_body}

                            resp = requests.post(issue_api_url, headers=headers, json=payload)

                            if resp.status_code == 201:
                                issue_url = resp.json().get("html_url", "")
                                st.success(f"✅ GitHub Issue Created: [View Issue]({issue_url})")
                            else:
                                st.error(f"❌ Failed to create issue. Status: {resp.status_code}")
                                st.code(resp.text)

                        except Exception as ex:
                            st.error(f"An error occurred while notifying GitHub: {ex}")

elif selected == "📁 Dataset Metadata":
    st.title("📁 Dataset Metadata")
    st.markdown("""
Review comprehensive metadata for the datasets used in your machine learning experiments.

📁 **What you’ll find**:
- Dataset titles, versions, and identifiers
- Authorship, publication dates, and publisher information
- Source platforms and linked repositories (e.g., DBRepo, Invenio)

🔍 **Why it matters**:
- Track the origin and integrity of training data  
- Ensure FAIR (Findable, Accessible, Interoperable, Reusable) data principles  
- Provide proper attribution in research outputs
""")
# 📁 Dataset Metadata (updated based on selected run)

    st.title("📁 Dataset Metadata")
    st.markdown("""
    Review comprehensive metadata for the datasets used in your machine learning experiments.
    """)
    
    # 1. Detect available runs
    run_ids = df['run_id'].dropna().unique()
    
    if not run_ids.any():
        st.warning("⚠️ No runs found. Please train a model first.")
    else:
        # 2. User selects a run
        selected_run = st.selectbox("Select a Run ID", run_ids)
    
        # 3. Filter the DataFrame for the selected run
        run_df = df[df["run_id"] == selected_run]
    
        # 4. Define relevant dataset metadata columns
        dataset_cols = [
            "param_dataset.title", "param_dataset.doi", "param_dataset.authors",
            "param_dataset.publisher", "param_dataset.published",
            "tag_dataset_id", "tag_dataset_name", "tag_dataset_version",
            "tag_data_source", "param_database.name", "param_database.owner",
            "tag_dbrepo.repository_name"
        ]
    
        # 5. Only keep existing columns
        dataset_cols = [c for c in dataset_cols if c in run_df.columns]
        dataset_info = run_df[dataset_cols]
    
        if not dataset_info.empty:
            st.write("### Selected Run - Dataset Metadata")
            st.dataframe(dataset_info.T, use_container_width=True)
        else:
            st.warning("No dataset metadata available for this run.")
    # # Filter relevant columns
    # dataset_cols = [
    #     "param_dataset.title", "param_dataset.doi", "param_dataset.authors",
    #     "param_dataset.publisher", "param_dataset.published",
    #     "tag_dataset_id", "tag_dataset_name", "tag_dataset_version",
    #     "tag_data_source", "param_database.name", "param_database.owner",
    #     "tag_dbrepo.repository_name"
    # ]

    # dataset_info = df[dataset_cols].copy()

    # # Dropdown to select dataset (optional, if more than one)
    # dataset_names = dataset_info["tag_dataset_name"].dropna().unique()

    # selected_dataset = st.selectbox("Choose dataset", dataset_names)

    # filtered_df = dataset_info[dataset_info["tag_dataset_name"] == selected_dataset]

    # if not filtered_df.empty:
    #     st.write("### Selected Dataset Metadata")
    #     st.dataframe(filtered_df.T)
    # else:
    #     st.warning("No matching dataset found.")


elif selected == "🧠 ML Model Metadata":
    st.title("🧠 ML Model Metadata")
    st.markdown("""
Explore detailed metadata about each machine learning model used in your experiments.

🧠 **What’s included**:
- Model hyperparameters (e.g., tree depth, split criteria)
- Training and test dataset configuration
- Python and ML library versions (e.g., scikit-learn, NumPy)
- Evaluation metrics such as accuracy, F1 score, ROC AUC

🔍 **Why it matters**:
- Validate model training conditions  
- Ensure consistent environments across experiments  
- Support reproducibility and audit readiness
""")
    run_ids = df['run_id'].dropna().unique()
    selected_run = st.selectbox("Select a Run ID", run_ids)
    run_df = df[df["run_id"] == selected_run]

    if run_df.empty:
        st.warning("No matching run found.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🛠️ Hyperparameters", "💻 Environment", "📊 Dataset Sampling", "📈 Metrics"
        ])

        with tab1:
            st.write("### Training Hyperparameters")
            st.dataframe(run_df.filter(
                regex=r'^param_(criterion|max_depth|max_features|min_samples_split|min_samples_leaf|n_estimators|bootstrap|warm_start|oob_score|random_state)'
            ).T)

        with tab2:
            st.write("### Environment Info")
            st.dataframe(run_df.filter(
                regex=r'^param_(numpy_version|pandas_version|python_version|sklearn_version|matplotlib_version|seaborn_version|os_platform)'
            ).T)

        with tab3:
            st.write("### Dataset Size & Sampling")
            st.dataframe(run_df.filter(
                regex=r'^param_(n_records|n_features|n_train_samples|n_test_samples|test_size|max_samples)'
            ).T)

        with tab4:
            st.write("### Model Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ✅ Evaluation (Test Set)")
                st.dataframe(run_df.filter(
                    regex=r'^metric_(accuracy$|f1_score_X_test|precision_score_X_test|recall_score_X_test|roc_auc_score_X_test)'
                ).T)

            with col2:
                st.markdown("#### 🏋️ Training Set")
                st.dataframe(run_df.filter(
                    regex=r'^metric_training_'
                ).T)

            st.markdown("#### 🛢️ DBRepo Lineage ")
            st.dataframe(run_df.filter(
                regex=r'^metric_dbrepo\..*'
            ).T)



elif selected == "📊 Model Plots":
    st.title("📊 Model Explainability & Evaluation Plots")
    st.markdown("""
Visualize how your machine learning model is performing — and understand **why** it's making the predictions it does.
""")

    # 1. Detect available plot folders
    plot_folders = glob.glob(os.path.join("plots", "RandomForest_Iris_v*"))
    plot_folders = [os.path.basename(folder) for folder in plot_folders]

    if not plot_folders:
        st.warning("⚠️ No plot folders found. Please run a training job first.")
    else:
        # 2. Let user pick which run
        selected_folder = st.selectbox("Select a Run (for plots)", plot_folders)

        # 3. Define plot options dynamically
        plot_options = {
            "Feature Importances": "feature_importances.png",
            "Confusion Matrix": "confusion_matrix.png",
            "SHAP Summary": "shap_summary.png",
            "ROC Curve (Class 0)": "roc_curve_cls_0.png",
            "Precision-Recall (Class 0)": "pr_curve_cls_0.png"
        }

        selected_plot = st.selectbox("Choose a Plot Type", list(plot_options.keys()))

        plot_path = os.path.join("plots", selected_folder, plot_options[selected_plot])

        if os.path.exists(plot_path):
            plot_width = st.slider("Adjust Plot Width", 400, 1000, 600)
            st.image(plot_path, caption=f"{selected_plot} — {selected_folder}", width=plot_width)

            if "Feature Importances" in selected_plot:
                st.markdown("**Interpretation:** Shows which features contribute most to predictions.")
            elif "SHAP" in selected_plot:
                st.markdown("**Interpretation:** SHAP summary plots show feature impact and distribution.")
            elif "ROC" in selected_plot:
                st.markdown("**Interpretation:** ROC curves visualize classifier trade-off between sensitivity and specificity.")
            elif "Precision-Recall" in selected_plot:
                st.markdown("**Interpretation:** Precision-Recall curves help understand classifier performance on imbalanced data.")
            elif "Confusion" in selected_plot:
                st.markdown("**Interpretation:** The confusion matrix shows how many predictions were correct or misclassified.")
        else:
            st.error("❌ Selected plot file not found!")



elif selected == "🛰️ Provenance Trace":
    st.title("🛰️ Provenance Trace")


    use_case_descriptions = {
    "trace_preprocessing": "🔍 Trace preprocessing steps for a run (e.g., dropped columns, features used).",
    "drop_impact": "📉 Measure accuracy impact of dropping a single feature.",
    "drop_impact_all": "🧪 Test each feature’s drop impact to assess global importance.",
    "best_feature_subset": "🎯 Evaluate model accuracy on a custom subset of features.",
    "common_high_accuracy": "🏆 Find preprocessing patterns in high-accuracy runs (above a threshold)."
}
    st.markdown("### 📘 Use Case Selector")
    for key, desc in use_case_descriptions.items():
        st.markdown(f"**`{key}`** – {desc}")
    
    # Use case selection
    use_case_name = st.selectbox(
        "Select a Use Case",
        options=list(USE_CASES.keys()),
        help="Choose an analysis utility to run on your ML provenance data."
    )
    	
    use_case = USE_CASES[use_case_name]
    
    # Collect required parameters
    params = {}
    for param in use_case['required_params']:
        if param == 'feature':
            all_features = _get_all_features(df)
            selected_feature = st.selectbox("Select a Feature", all_features)
            params['feature'] = selected_feature
        elif param == 'features':
            all_features = _get_all_features(df)
            selected_features = st.multiselect("Select Features", all_features)
            params['features'] = selected_features
        elif param == 'threshold':
            threshold = st.slider("Set Accuracy Threshold", min_value=0.0, max_value=1.0, value=0.95)
            params['threshold'] = threshold
        else:
            param_value = st.text_input(f"Enter value for {param}")
            params[param] = param_value

    # Collect optional parameters
    for param in use_case['optional_params']:
        if param == 'run_id':
            run_ids = df['run_id'].dropna().unique()
            selected_run_id = st.selectbox("Select a Run ID (Optional)", run_ids)
            params['run_id'] = selected_run_id
        else:
            param_value = st.text_input(f"Enter value for {param} (Optional)")
            params[param] = param_value

    # Execute the selected use case
    if st.button("Run Use Case"):
        try:
            result = use_case['func'](df, **params)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif isinstance(result, list):
                st.json(result)
            elif isinstance(result, dict):
                st.json(result)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")



elif selected == "🧭 Model-Dataset Mapping":
    st.title("🧭 Model-Dataset Mapping")
    st.markdown("""
Gain insights into which machine learning models were trained on which datasets — and view the associated metadata.

🔗 **What you’ll see**:
- Model names used in experiments
- Dataset titles, DOIs, publishers, and version info
- Attribution-ready links for research transparency

🧪 **Why it's useful**:
- Ensure proper dataset-model pairing  
- Validate that models were trained on published or approved datasets  
- Maintain clear provenance and reproducibility

""")   

    try:
        results = map_model_dataset(df)
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("No model-dataset mappings found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

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
    
    🧠 These justifications help ensure **transparency**, **explainability**, and support for reproducibility.
    """)

   # 1. Detect available MODEL_PROVENANCE folders
    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]
    
    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        # 2. Let user pick the run
        selected_provenance_folder = st.selectbox("Select a Run (for Justifications)", provenance_folders)
    
        # 3. Build path to justification file
        justification_file = os.path.join(
            "MODEL_PROVENANCE",
            selected_provenance_folder,
            f"{selected_provenance_folder}_run_summary.json"
        )
    
        try:
            df_just = load_justification_table(justification_file)
            st.success(f"Loaded: `{justification_file}`")
            st.write("### Justification Table")
            st.dataframe(df_just, use_container_width=True)
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
