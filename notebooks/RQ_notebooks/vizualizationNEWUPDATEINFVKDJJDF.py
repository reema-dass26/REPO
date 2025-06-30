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
from streamlit_option_menu import option_menu
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import time
from datetime import datetime
import re
from pathlib import Path
st.set_page_config(
    page_title="Building Bridges in Research: Integrating Provenance and Data Management in Virtual Research Environments",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

############################################################################
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

from rdflib import Graph
from rdflib.namespace import RDF, DC, DCTERMS, FOAF, Namespace
from rdflib.term import BNode
from pyvis.network import Network
import os

def visualize_interactive_provenance(rdf_path, html_output_path):
    g = Graph()
    g.parse(rdf_path)

    # Set up PyVis network
    net = Network(height="750px", width="100%", directed=True)
    net.force_atlas_2based()

    # Optional: your base namespace
    EX = Namespace("https://github.com/reema-dass26/ml-provenance/provenance/")

    def get_label(node):
        """Return a clean, readable label or None if system-generated."""
        for prop in [DC.title, DCTERMS.title, FOAF.name, EX["modelName"]]:
            label = g.value(subject=node, predicate=prop)
            if label:
                return str(label)
        node_str = str(node)
        fallback = node_str.split("/")[-1]
        if "example.org" in node_str or "example.com" in node_str or fallback.startswith("Node") or fallback.startswith("genid") or len(fallback) < 2:
            return None
        return fallback

    for subj, pred, obj in g:
        # Skip RDF noise
        if pred == RDF.type:
            continue

        # Skip system-generated blank nodes
        if isinstance(subj, BNode) or isinstance(obj, BNode):
            continue

        # Skip if URI contains example.org/com
        if "example.org" in str(subj) or "example.com" in str(subj) or \
           "example.org" in str(obj) or "example.com" in str(obj):
            continue

        # Try to get clean labels
        subj_label = get_label(subj)
        obj_label = get_label(obj)
        pred_label = str(pred).split("/")[-1]

        # Skip if labels are not clean
        if not subj_label or not obj_label:
            continue

        net.add_node(str(subj), label=subj_label)
        net.add_node(str(obj), label=obj_label)
        net.add_edge(str(subj), str(obj), label=pred_label)

    os.makedirs(os.path.dirname(html_output_path), exist_ok=True)
    net.show(html_output_path)



import os
import glob
import json
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """
    Load and compile all structured_metadata_final.json files from MODEL_PROVENANCE.
    Returns a DataFrame with one row per run.
    """
    pattern = os.path.join("MODEL_PROVENANCE", "*", "structured_metadata.json")
    files = glob.glob(pattern)

    if not files:
        st.warning("⚠️ No structured metadata files found.")
        return pd.DataFrame()

    rows = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                structured = json.load(f)

            # Combine run ID from folder
            folder = os.path.basename(os.path.dirname(file_path))
            flat_row = {"run_id": folder}

            # Flatten each category
            for section, section_data in structured.items():
                if isinstance(section_data, dict):
                    for key, val in section_data.items():
                        flat_row[f"{section}_{key}"] = val
                else:
                    flat_row[section] = section_data

            rows.append(flat_row)
            print(f"✅ Loaded: {folder}")

        except Exception as e:
            st.error(f"❌ Failed to load {file_path}: {e}")
    
    df = pd.DataFrame(rows)
    st.success(f"✅ Loaded {len(df)} structured runs with {len(df.columns)} fields.")
    print(df.columns.tolist())
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
            # "📚 Invenio Metadata",
            "⚙️ Environment Requirements"

            
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
   

    st.markdown("""<style>
    body, .main {
        background-color: #121212;
        color: #e0e0e0;
    }

    .block-container {
        padding: 2rem;
        max-width: 1400px;
    }

    div[data-testid="column"] > div {
        background-color: #1f1f1f;
        padding: 1.2rem 1rem;
        margin: 0.8rem;
        border-radius: 0.6rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, background-color 0.3s ease;
        min-height: 180px;
    }

    div[data-testid="column"] > div:hover {
        background-color: #2c2c2c;
        transform: translateY(-4px);
    }

    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        margin-top: 0;
    }

    p {
        color: #cccccc;
        font-size: 0.95rem;
        line-height: 1.5rem;
    }

    .dashboard-title {
        margin-bottom: 2rem;
        margin-top: 1rem;
    }

    footer {
        visibility: hidden;
    }
</style>""", unsafe_allow_html=True)


    # Title
    st.markdown("## 👋 Welcome to the End-to-End Provenance Dashboard")
    
    # Section metadata
    sections = [
        {"emoji": "🧬", "title": "Dataset Metadata", "desc": "Authorship, DOIs, transformations, and links to DBRepo."},
        {"emoji": "🧠", "title": "ML Model Metadata", "desc": "Architecture, hyperparameters, training setup, and compute logs."},
        {"emoji": "📊", "title": "Model Plots", "desc": "SHAP, ROC, PR curves, confusion matrices with metadata links."},
        {"emoji": "🛰️", "title": "Provenance Trace", "desc": "Reconstruct training paths using data, code, parameters, and preprocessing."},
        {"emoji": "🧨", "title": "Error & Version Impact", "desc": "Detect deprecated runs and notify researchers of faulty configurations."},
        {"emoji": "🧭", "title": "Model–Dataset Mapping", "desc": "Cross-link models and datasets to validate provenance and consistency."},
        {"emoji": "📘", "title": "Researcher Justifications", "desc": "Log rationale behind modeling decisions for transparency."},
        {"emoji": "📣", "title": "Notify Fork Owners", "desc": "Alert GitHub users with outdated forks using auto-generated issues."},
        {"emoji": "📤", "title": "Export Metadata", "desc": "Export structured metadata (YAML, JSON, PROV-O) for archival or publication."},
        # {"emoji": "📚", "title": "Invenio Metadata", "desc": "Render Invenio-style metadata records for datasets and publications."}
    ]
    
    # Dynamically create rows of 3 columns each
    for i in range(0, len(sections), 3):
        cols = st.columns(3)
        for col, section in zip(cols, sections[i:i+3]):
            with col:
                st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem;
                                box-shadow: 0 0 10px rgba(0, 0, 0, 0.3); transition: 0.3s ease-in-out;">
                        <h4 style="margin-bottom: 0.5rem;">{section['emoji']} {section['title']}</h4>
                        <p style="font-size: 0.9rem; color: #d0d0d0;">{section['desc']}</p>
                    </div>
                """, unsafe_allow_html=True)


    st.markdown("---")
    st.info("🔍 Use the **sidebar** to navigate to each section. This dashboard supports RQ1–RQ4 through deep metadata inspection and provenance visualization.")

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
            # st.balloons()
    
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
        st.warning("⚠️ No runs found.")
    else:
        selected_run = st.selectbox("Select a Run ID", run_ids)
        run_df = df[df["run_id"] == selected_run]

        if run_df.empty:
            st.warning("No metadata available.")
            st.stop()

        row = run_df.iloc[0].to_dict()

        # Display label formatting function
        def format_label(k):
            if ":" in k:
                ns, name = k.split(":", 1)
                return f"{ns.upper()} {name.replace('_', ' ').capitalize()}"
            return k.replace("_", " ").capitalize()

        def show_table(title, keys):
            st.subheader(title)
            table = {}
            for k in keys:
                val = row.get(k)
                label = format_label(k)
                table[label] = val if pd.notna(val) and val != "" else "—"
            if table:
                st.dataframe(pd.DataFrame(list(table.items()), columns=["Field", "Value"]), use_container_width=True)
            else:
                st.info("ℹ️ No data available for this section.")

       # Filter all FAIR keys first
        fair_keys = [k for k in row if k.startswith(("FAIR_dc:", "FAIR_dcterms:", "FAIR_dcat:", "FAIR_"))]
        
        # Prepare display table with only keys that have real values
        table = {}
        for k in fair_keys:
            val = row.get(k)
            if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
                continue  # skip empty/missing
            label = k.split(":", 1)[-1].replace("_", " ").capitalize()  # simple label
            if label not in table:  # prefer first populated key to avoid duplicates
                table[label] = val
        
        if table:
            st.dataframe(pd.DataFrame(list(table.items()), columns=["Field", "Value"]), use_container_width=True)
        else:
            st.info("ℹ️ No FAIR dataset metadata available.")

        provo_keys = [k for k in row if k.startswith("PROV-O_prov:")]
        if any(
            val is not None and not (isinstance(val, float) and pd.isna(val)) and val != ""
            for val in (row.get(k) for k in provo_keys)
        ):
            show_table("🛰️ PROV-O Provenance Metadata", provo_keys)
        else:
            st.info("ℹ️ No provenance metadata available for this run.")

        # 🧪 Preprocessing Info
        st.subheader("🧪 Preprocessing Info")
        prep_info = row.get("Croissant_mls:preprocessingSteps", row.get("Uncategorized_preprocessing_info"))
        try:
            if isinstance(prep_info, str):
                prep_info = json.loads(prep_info)
            elif prep_info is None:
                prep_info = {}

            prep_rows = []
            for k, v in prep_info.items():
                pretty_val = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
                prep_rows.append({
                    "Step": k.replace("_", " ").capitalize(),
                    "Details": pretty_val
                })

            if prep_rows:
                st.dataframe(pd.DataFrame(prep_rows), use_container_width=True)
            else:
                st.info("ℹ️ No preprocessing info found.")
        except Exception as e:
            st.warning(f"⚠️ Could not parse preprocessing info: {e}")



    
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

    import subprocess
    import json

    df["_git_commit_hash"] = df["FAIR4ML_fair4ml:trainingScriptVersion"].fillna("—")
    df["_git_version"] = df["FAIR_dcterms:hasVersion"].fillna("untagged")

    # Get current git commit hash from local repo (if applicable)
    def get_current_git_commit():
        import subprocess
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except Exception:
            return None
    
    current_hash = get_current_git_commit()
    
    # Map commit hash → version tag for lookup
    version_map = df.set_index("_git_commit_hash")["_git_version"].to_dict()
    current_version_tag = version_map.get(current_hash, "untagged")
    
    st.markdown("### 🏷️ Git Commit – Version Mapping")
    
    if current_hash:
        st.markdown(f"### 📌 Current Git Commit: `{current_hash}`")
    
    # Show a dataframe with selected columns including flat keys
    st.dataframe(
        df[[
            "FAIR4ML_fair4ml:runID",
            "PROV-O_prov:commit",
            "FAIR_dcterms:hasVersion",
            "_git_commit_hash",
            "_git_version"
        ]],
        use_container_width=True
    )

    
    # # Input deprecated versions to flag experiments
    # deprecated_versions_input = st.text_area("Enter deprecated version tags (one per line):", height=100)
    # simulate_current = st.checkbox("☢️ Also mark current local commit as deprecated")
    
    # if simulate_current and current_version_tag:
    #     deprecated_versions_input += f"\n{current_version_tag}"
    #     st.info(f"☢️ Added current version `{current_version_tag}` to deprecated list.")
    
    # deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]
    
    # def detect_deprecated_versions(df, deprecated_versions):
    #     # Use the parsed _git_version column to find affected runs
    #     affected = df[df["_git_version"].isin(deprecated_versions)].copy()
    #     if "Uncategorized_git_metadata" in df.columns:
    #         git_data = df["Uncategorized_git_metadata"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    #         affected["github_user"] = git_data.apply(
    #             lambda g: g.get("author_email", "").split("+")[-1].split("@")[0] if "author_email" in g else None
    #         )
    #     return affected

    # if st.button("🚨 Detect Impacted Runs"):
    #     if not deprecated_versions:
    #         st.warning("Please enter at least one deprecated version.")
    #     else:
    #         detected = detect_deprecated_versions(df, deprecated_versions)
    #         if detected.empty:
    #             st.success("✅ No impacted runs found.")
    #         else:
    #             st.warning("⚠️ Impacted Experiments Detected:")
    #             st.session_state.results_df = detected
    #             st.dataframe(st.session_state.results_df, use_container_width=True)

    # if not st.session_state.results_df.empty:
    #     st.markdown("### 📣 Notify Affected Users via GitHub")
    #     with st.expander("🔐 GitHub Authentication"):
    #         owner = st.text_input("GitHub Owner", value="reema-dass26")
    #         repo = st.text_input("Repository Name", value="REPO")
    #         token = st.text_input("GitHub Token", type="password")

    #         if st.button("📬 Notify Affected Users"):
    #             if not all([owner, repo, token]):
    #                 st.warning("❗ Provide all GitHub credentials.")
    #             else:
    #                 try:
    #                     impacted_users = st.session_state.results_df["prov:commitEmail"].dropna().unique()
    #                     user_tags = " ".join(f"@{u}" for u in impacted_users if u)
    #                     issue_body = (
    #                         f"The following experiments were run on deprecated versions:\n\n"
    #                         f"- Versions: {', '.join(set(deprecated_versions)) or 'N/A'}\n\n"
    #                         f"{user_tags or '—'}\n\n"
    #                         "Please retrain or validate your experiments.\n\n"
    #                         "— Provenance Dashboard"
    #                     )

    #                     headers = {
    #                         "Authorization": f"token {token}",
    #                         "Accept": "application/vnd.github+json"
    #                     }
    #                     issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    #                     resp = requests.post(issue_url, headers=headers, json={"title": "🚨 Deprecated Version Used", "body": issue_body})

    #                     if resp.status_code == 201:
    #                         st.success(f"✅ GitHub Issue Created: [View Issue]({resp.json().get('html_url')})")
    #                     else:
    #                         st.error(f"❌ GitHub Issue failed: {resp.status_code}")
    #                         st.code(resp.text)
    #                 except Exception as e:
    #                     st.error(f"Exception occurred: {str(e)}")

    # import json
    # deprecated_versions_input = st.text_area("Enter deprecated version tags (one per line):", height=100)

    # deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]

    # def extract_commit_email(metadata):
    #     if not metadata:
    #         return None
    #     try:
    #         data = json.loads(metadata) if isinstance(metadata, str) else metadata
    #         return data.get("PROV-O", {}).get("prov:commitEmail", None)
    #     except Exception:
    #         return None
    #      "PROV-O_prov:commit",         # flat commit hash from provenance
    #         "FAIR_dcterms:hasVersion"
    # def extract_github_username(email):
    #     if not email or "+" not in email:
    #         return None
    #     try:
    #         return email.split("+")[1].split("@")[0]
    #     except IndexError:
    #         return None
    
    # def detect_deprecated_versions(df, deprecated_versions):
    #     affected = df[df["FAIR_dcterms:hasVersion"].isin(deprecated_versions)].copy()
    
    #     if "PROVO" in affected.columns:
    #         affected["prov:commitEmail"] = affected["PROV-O_prov:commitEmail"].apply(extract_commit_email)
    #     else:
    #         affected["prov:commitEmail"] = None
    
    #     return affected
    
    # # ... your existing UI code ...
    
    # if st.button("🚨 Detect Impacted Runs"):
    #     if not deprecated_versions:
    #         st.warning("Please enter at least one deprecated version.")
    #     else:
    #         detected = detect_deprecated_versions(df, deprecated_versions)
    #         if detected.empty:
    #             st.success("✅ No impacted runs found.")
    #         else:
    #             st.warning("⚠️ Impacted Experiments Detected:")
    #             st.session_state.results_df = detected
    #             st.dataframe(st.session_state.results_df, use_container_width=True)
    
    # if not st.session_state.results_df.empty:
    #     st.markdown("### 📣 Notify Affected Users via GitHub")
    #     with st.expander("🔐 GitHub Authentication"):
    #         owner = st.text_input("GitHub Owner", value="reema-dass26")
    #         repo = st.text_input("Repository Name", value="REPO")
    #         token = st.text_input("GitHub Token", type="password")
    
    #         if st.button("📬 Notify Affected Users"):
    #             if not all([owner, repo, token]):
    #                 st.warning("❗ Provide all GitHub credentials.")
    #             else:
    #                 try:
    #                     impacted_emails = st.session_state.results_df["prov:commitEmail"].dropna().unique()
    #                     impacted_users = [extract_github_username(e) for e in impacted_emails if e]
    #                     user_tags = " ".join(f"@{u}" for u in impacted_users if u)
                        
    #                     issue_body = (
    #                         f"The following experiments were run on deprecated versions:\n\n"
    #                         f"- Versions: {', '.join(set(deprecated_versions)) or 'N/A'}\n\n"
    #                         f"{user_tags or '—'}\n\n"
    #                         "Please retrain or validate your experiments.\n\n"
    #                         "— Provenance Dashboard"
    #                     )
    
    #                     headers = {
    #                         "Authorization": f"token {token}",
    #                         "Accept": "application/vnd.github+json"
    #                     }
    #                     issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    #                     resp = requests.post(issue_url, headers=headers, json={"title": "🚨 Deprecated Version Used", "body": issue_body})
    
    #                     if resp.status_code == 201:
    #                         st.success(f"✅ GitHub Issue Created: [View Issue]({resp.json().get('html_url')})")
    #                     else:
    #                         st.error(f"❌ GitHub Issue failed: {resp.status_code}")
    #                         st.code(resp.text)
    #                 except Exception as e:
    #                     st.error(f"Exception occurred: {str(e)}")
    import json
    import requests  # ensure you import requests
    
    # Deprecated versions input box, provide placeholder as string
    deprecated_versions_input = st.text_area(
        "Enter deprecated version tags (one per line):",
        height=100,
        placeholder="e.g. v1.0.0\nv2.0.3\nabc123commit"
    )
    
    deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]
    
    def extract_commit_email(metadata):
        if not metadata:
            return None
        try:
            # metadata could be JSON string or dict
            data = json.loads(metadata) if isinstance(metadata, str) else metadata
            # According to your data structure, PROV-O might be a dict inside
            return data.get("PROV-O", {}).get("prov:commitEmail", None) or data.get("prov:commitEmail")
        except Exception:
            return None
    
    def extract_github_username(email):
        if not email or "+" not in email:
            return None
        try:
            # Extract username after '+' and before '@'
            return email.split("+")[1].split("@")[0]
        except IndexError:
            return None
    
    def detect_deprecated_versions(df, deprecated_versions):
        # Filter rows where version column matches deprecated versions
        affected = df[df["FAIR_dcterms:hasVersion"].isin(deprecated_versions)].copy()
    
        # Check if 'PROV-O_prov:commitEmail' column exists and extract emails
        if "PROV-O_prov:commitEmail" in affected.columns:
            affected["prov:commitEmail"] = affected["PROV-O_prov:commitEmail"].apply(extract_commit_email)
        else:
            # Fall back: try to extract from 'Uncategorized_git_metadata' if available
            if "Uncategorized_git_metadata" in affected.columns:
                affected["prov:commitEmail"] = affected["Uncategorized_git_metadata"].apply(extract_commit_email)
            else:
                affected["prov:commitEmail"] = None
    
        return affected
    
    # UI trigger to detect deprecated runs
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
    
    # Notify affected users via GitHub
    if "results_df" in st.session_state and not st.session_state.results_df.empty:
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
                        impacted_emails = st.session_state.results_df["prov:commitEmail"].dropna().unique()
                        impacted_users = [extract_github_username(e) for e in impacted_emails if e]
                        user_tags = " ".join(f"@{u}" for u in impacted_users if u)
                        
                        issue_body = (
                            f"The following experiments were run on deprecated versions:\n\n"
                            f"- Versions: {', '.join(set(deprecated_versions)) or 'N/A'}\n\n"
                            f"{user_tags or '—'}\n\n"
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


elif selected == "🧠 ML Model Metadata":
    st.title("🧠 ML Model Metadata")

    run_ids = df['run_id'].dropna().unique()
    if not run_ids.any():
        st.warning("⚠️ No runs found.")
    else:
        selected_run = st.selectbox("Select a Run ID", run_ids)
        run_df = df[df["run_id"] == selected_run]

        if run_df.empty:
            st.warning("No metadata available for this run.")
            st.stop()

        row = run_df.iloc[0].to_dict()

        def clean(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val, indent=2)
            return str(val) if val else "—"

        def show_section(title, data):
            st.subheader(title)
            df = pd.DataFrame(
                [{"Field": k, "Value": clean(v)} for k, v in data.items()]
            )
            st.dataframe(df, use_container_width=True)

        # 🚀 Model Overview
        # 🚀 Overview from Croissant
        show_section("🚀 Model Overview", {
            "Model Name": row.get("Croissant_mls:modelName", "—"),
            "Algorithm": row.get("Croissant_mls:learningAlgorithm", "—"),
            "Architecture": row.get("Croissant_mls:modelArchitecture", "—"),
            "Serialization Format": row.get("Croissant_mls:serializationFormat", "—"),
            "Target Variable": row.get("Croissant_mls:targetVariable", "—"),
            "Label Encoding": row.get("Croissant_mls:labelEncoding", "—"),
            "Model Path": row.get("Croissant_mls:modelPath", "—"),
            "Model Version": row.get("Croissant_mls:modelVersion", "—"),
        })


        # 🧠 Hyperparameters
        hyper = row.get("Croissant_mls:hyperparameters", {})
        if isinstance(hyper, str):
            try:
                hyper = json.loads(hyper)
            except Exception:
                hyper = {"error": "Could not parse hyperparameters"}
        show_section("🧠 Hyperparameters", hyper)

        import numpy as np

        def safe_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return val
        
        # Filter keys for relevant metrics (case-insensitive)
        metric_keys = [
            k for k in row.keys() 
            if k.startswith("MLSEA_") and any(m in k.lower() for m in ["accuracy", "f1", "precision", "recall", "roc"])
        ]
        import numpy as np
        # Prepare display dict with cleaned keys and safe values
        def safe_val(val):
            
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return val
        
        excluded_prefixes = ["mlsea_justification_"]

        test_metric_keys = [
            k for k in row.keys()
            if k.lower().startswith("mlsea_")
            and not any(k.lower().startswith(prefix) for prefix in excluded_prefixes)
            and any(m in k.lower() for m in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc","f1_score","precision","recall"])
            and "training" not in k.lower()
        ]
        def is_valid_metric(value):
            return value not in [None, "", "—", "No justification provided"]
        
        # Build dict of key: value pairs from row using your filtered keys
        metrics = {k: row[k] for k in test_metric_keys[0:5]}
        
        # Now filter out invalid metrics
        cleaned_metrics = {k: v for k, v in metrics.items() if is_valid_metric(v)}
        print(cleaned_metrics)



        
        training_metric_keys = [
            k for k in row.keys()
            if k.startswith("MLSEA_") and 
               any(m in k.lower() for m in ["training_accuracy", "training_f1", "training_precision", "training_recall", "training_roc"]) and
               "training" in k.lower()
        ]
        
        def clean_key(k):
            # Remove prefixes and underscores, capitalize nicely
            k = k.replace("MLSEA_mlsea:", "").replace("MLSEA_justification_", "").replace("_", " ")
            return k.capitalize()
        
        test_metrics = {clean_key(k): safe_val(row[k]) for k in cleaned_metrics}
        training_metrics = {clean_key(k): safe_val(row[k]) for k in training_metric_keys}
        
        show_section("📊 Test Metrics", test_metrics)
        show_section("📊 Training Metrics", training_metrics)






elif selected == "📊 Model Plots":
    st.title("📊 Model Explainability & Evaluation Plots")
    st.markdown("""
Visualize how your machine learning model is performing — and understand **why** it's making the predictions it does.

🔗 This section links each plot back to the run folder and summary metadata.
""")

    import glob
    import json
    import os
    import pandas as pd

    # Step 1: Find all run folders with a summary JSON
    folder_paths = glob.glob(os.path.join("MODEL_PROVENANCE", "*", "*_run_summary.json"))
    run_id_to_folder = {}

    for path in folder_paths:
        folder = os.path.dirname(path)
        folder_name = os.path.basename(folder)
        run_id_to_folder[folder_name] = folder

    if not run_id_to_folder:
        st.warning("⚠️ No run folders with summary JSONs found.")
        st.stop()

    # Step 2: Let user select a run folder (run_id = folder name)
    selected_run = st.selectbox("Select a Run ID", sorted(run_id_to_folder.keys()))
    run_folder = run_id_to_folder[selected_run]

    st.success(f"📁 Loaded run folder: `{selected_run}`")

    # Step 3: Load summary JSON
    summary_path = glob.glob(os.path.join(run_folder, "structured_metadata.json"))
    if not summary_path:
        st.error("❌ Could not find a summary JSON file in the selected folder.")
        st.stop()

    with open(summary_path[0], "r") as f:
        run_data = json.load(f)

    # ── Extended Metadata ──
    with st.expander("📋 Extended Metadata"):

        def safe_str(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            elif val is None:
                return "—"
            return str(val)
        meta_preview = {
    "Run ID": selected_run,
    "Model Name": run_data.get("Croissant", {}).get("mls:modelName", "—"),
    "Algorithm": run_data.get("Croissant", {}).get("mls:learningAlgorithm", "—"),
    "Architecture": run_data.get("Croissant", {}).get("mls:modelArchitecture", "—"),
    "Serialization Format": run_data.get("Croissant", {}).get("mls:serializationFormat", "—"),
    "Model Path": run_data.get("Croissant", {}).get("mls:modelPath", "—"),
    "Model Version": run_data.get("Croissant", {}).get("mls:modelVersion", "—"),
    "Target Variable": run_data.get("Croissant", {}).get("mls:targetVariable", "—"),
    "Label Encoding": run_data.get("Croissant", {}).get("mls:labelEncoding", "—"),

    "Dataset Title": run_data.get("FAIR", {}).get("dc:title", "—"),
    "Training Start": run_data.get("FAIR4ML", {}).get("fair4ml:trainingStartTime", "—"),
    "Training End": run_data.get("FAIR4ML", {}).get("fair4ml:trainingEndTime", "—"),
    "Accuracy (Test)": run_data.get("MLSEA", {}).get("mlsea:accuracy", "—"),
}

# Add MLSEA metrics only if valid
        test_metric_keys = [
            k for k in run_data.keys()
            if k.lower().startswith("mlsea_")
            and not any(k.lower().startswith(prefix) for prefix in excluded_prefixes)
            and any(m in k.lower() for m in ["accuracy", "f1", "precision", "recall", "roc"])
            and "training" not in k.lower()
        ]
        def is_valid_metric(value):
            return value not in [None, "", "—", "No justification provided"]
        
        # Build dict of key: value pairs from row using your filtered keys
        metrics = {k: row[k] for k in test_metric_keys[0:5]}
        
        # Now filter out invalid metrics
        cleaned_metrics = {k: v for k, v in metrics.items() if is_valid_metric(v)}
    
        training_metric_keys = [
            k for k in run_data.keys()
            if k.startswith("MLSEA_") and 
               any(m in k.lower() for m in ["accuracy", "f1", "precision", "recall", "roc"]) and
               "training" in k.lower()
        ]
        
        def clean_key(k):
            # Remove prefixes and underscores, capitalize nicely
            k = k.replace("MLSEA_mlsea:", "").replace("MLSEA_justification_", "").replace("_", " ")
            return k.capitalize()
        
        test_metrics = {clean_key(k): safe_val(row[k]) for k in cleaned_metrics}
        training_metrics = {clean_key(k): safe_val(row[k]) for k in training_metric_keys}
        
        meta_preview.update(test_metrics)
        meta_preview.update(training_metrics)



        
     
        # Optional Hyperparameters
        try:
            hparams = json.loads(run_data.get("tags_hyperparameters", "{}"))
        except:
            hparams = {}

        for k, v in hparams.items():
            meta_preview[f"Hyperparam → {k}"] = v

        # Optional Preprocessing Info
        try:
            prep = json.loads(run_data.get("tags_preprocessing_info", "{}"))
        except:
            prep = {}

        for k in ["dropped_columns", "final_feature_columns", "target_column"]:
            if k in prep:
                meta_preview[f"Preprocessing → {k}"] = prep[k]

        cleaned = {k: safe_str(v) for k, v in meta_preview.items()}
        st.dataframe(pd.DataFrame(list(cleaned.items()), columns=["Field", "Value"]), use_container_width=True)

    # ── Plot Viewer ──
    st.markdown("### 📈 Select and View Plot")

    plot_files = glob.glob(os.path.join(run_folder, "*.png"))

    if not plot_files:
        st.warning("⚠️ No plots found in the selected run folder.")
        st.stop()

    plot_options = {}
    for fpath in plot_files:
        fname = os.path.basename(fpath).replace(".png", "")
        label = fname.replace("_", " ").title()
        plot_options[label] = fpath

    selected_plot = st.selectbox("Choose Plot", list(plot_options.keys()))
    plot_path = plot_options[selected_plot]

    plot_width = st.slider("Adjust Plot Width", 400, 1000, 600)
    st.image(plot_path, caption=f"{selected_plot} — {selected_run}", width=plot_width)

    # ── Interpretation ──
    explanations = {
        "Feature Importances": "Shows which features contribute most to predictions.",
        "Shap Summary": "SHAP values show feature impact and distribution.",
        "Roc Curve": "Visualizes true vs. false positive rates.",
        "Precision Recall": "Helps evaluate classifier performance under class imbalance.",
        "Confusion Matrix": "Compares predicted vs. actual outcomes."
    }

    for key, explanation in explanations.items():
        if key.lower() in selected_plot.lower():
            st.markdown(f"**Interpretation:** {explanation}")
            break


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
        mapping_records = []
        print("dddddddddddddddddddddddddddddddddddddddddddddddd")
        print(df.columns.tolist())

        for _, row in df.iterrows():
            model_name = row.get("Croissant_mls:modelName", "—")
            architecture = row.get("Croissant_mls:modelArchitecture", "—")
            dataset_title = row.get("FAIR_dc:title", "—")
            dataset_version = row.get("FAIR_dcterms:hasVersion", "—")
            dataset_url = row.get("FAIR_dcat:landingPage", "—")
            accuracy = row["MLSEA_mlsea:accuracy"] if "MLSEA_mlsea:accuracy" in row else "—"
            # recall_macro = row["MLSEA_mlsea:recall_macro"] if "MLSEA_mlsea:recall_macro" in row else "—"
            # precision_macro = row["MLSEA_mlsea:precision_macro"] if "MLSEA_mlsea:precision_macro" in row else "—"

            roc_auc = row.get("MLSEA_mlsea:roc_auc", "—")
            run_id = row.get("run_id", "—")

            mapping_records.append({
                "Run ID": run_id,
                "Model Name": model_name,
                "Architecture": architecture,
                "Dataset Title": dataset_title,
                "Dataset Version": dataset_version,
                "Dataset Access URL": dataset_url,
                "Accuracy (Test)": accuracy,
                # "Recall (Test)": recall_macro,
                # "Precision (Test)": precision_macro,

                "ROC AUC (Test)": roc_auc
            })
        print( mapping_records)

        if mapping_records:
            df_mapping = pd.DataFrame(mapping_records)
            st.dataframe(df_mapping, use_container_width=True)
        else:
            st.warning("⚠️ No valid model-dataset mappings found.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")


elif selected == "🛰️ Provenance Trace":
    st.title("🛰️ Provenance Trace")
    st.markdown("""
Use this view to inspect detailed provenance metadata for a specific training run — and optionally compare it with another.

📌 **Use Case Highlights**:
- **Provenance & Reproducibility**: Trace how a model result was produced — including versions, parameters, and preprocessing.
- **Training Configuration & Evaluation**: Compare hyperparameters, strategies, and performance outcomes across runs.
""")
      # 🔧 Convert flattened row to nested structure
    def rebuild_nested_from_flat(flat_row):
        nested = {
            "Croissant": {},
            "FAIR": {},
            "FAIR4ML": {},
            "MLSEA": {},
            "PROV-O": {},
            "run_id": flat_row.get("run_id", "—")
        }
    
        for key, val in flat_row.items():
            if key.startswith("Croissant_"):
                nested["Croissant"][key.replace("Croissant_", "")] = val
            elif key.startswith("FAIR_"):
                nested["FAIR"][key.replace("FAIR_", "")] = val
            elif key.startswith("FAIR4ML_"):
                nested["FAIR4ML"][key.replace("FAIR4ML_", "")] = val
            elif key.startswith("MLSEA_"):
                nested["MLSEA"][key.replace("MLSEA_", "")] = val
            elif key.startswith("PROV-O_"):
                nested["PROV-O"][key.replace("PROV-O_", "")] = val
    
        return nested


    run_ids = df['run_id'].dropna().unique()  # 🔧 ADD THIS LINE

    selected_run = st.selectbox("Select Run 1", run_ids)
    # 📥 Download Reproducibility Guide
    repro_path = Path("MODEL_PROVENANCE") / selected_run / f"{selected_run}_reproducibility.txt"
    
    if repro_path.exists():
        with open(repro_path, "r", encoding="utf-8") as file:
            reproducibility_content = file.read()
    
        st.markdown("### 📥 Download Reproducibility Guide")
        st.download_button(
            label="⬇️ Download Reproducibility Instructions",
            data=reproducibility_content,
            file_name=f"{selected_run}_reproducibility.txt",
            mime="text/plain"
        )
    else:
        st.info("ℹ️ No reproducibility guide found for this run.")

    run_data_1 = df[df['run_id'] == selected_run].iloc[0].to_dict()
    run_data_1 = rebuild_nested_from_flat(run_data_1)  # <-- ADD THIS
    
    compare_mode = st.checkbox("🔁 Compare with another run")
    run_data_2 = None
    second_run = None
  
    if compare_mode:
        second_run = st.selectbox("Select Run 2", [r for r in run_ids if r != selected_run])
        run_data_2_df = df[df['run_id'] == second_run]
        if not run_data_2_df.empty:
            run_data_2 = run_data_2_df.iloc[0].to_dict()
            run_data_2 = rebuild_nested_from_flat(run_data_2)  # <-- AND THIS

    
  # ✅ Function to extract provenance fields (updated with correct keys and parsed fields)
    def get_provenance_fields(run_data):
        print(run_data.get("PROV-O_prov:commit", "—"))
        croissant = run_data.get("Croissant", {})
        fair = run_data.get("FAIR", {})
        prov = run_data.get("PROV-O", {})
        fair4ml = run_data.get("FAIR4ML", {})
        uncategorized = run_data.get("Uncategorized", {})
    
        # ✅ Parse preprocessing info string
        try:
            preprocessing_info = json.loads(croissant.get("mls:preprocessingSteps", "{}"))
        except Exception:
            preprocessing_info = {}
    
        return {
            "Run ID": run_data.get("run_id", "—"),
            "Dataset Title": fair.get("dc:title", "—"),
            "Dataset Version": fair.get("dcterms:hasVersion", "—"),
            "Dataset Source URL": fair.get("dcat:landingPage", "—"),
    
            # ✅ Fixed key for notebook name
            "Notebook Name": fair4ml.get("fair4ml:usedNotebook", {}),
    
            "Model Path": croissant.get("mls:modelPath", "—"),
            "Model Architecture": croissant.get("mls:modelArchitecture", "—"),
    
    
            # ✅ Fixed commit hash + fallback URL
            "Git Commit Hash": prov.get("prov:commit", "—"),
            "Git Commit Author": prov.get("prov:commitAuthor", "—"),
            "Git Branch": prov.get("prov:branch", "—"),
            "Git Commit Time": prov.get("prov:commitTime", "—"),
            "Git Repository": prov.get("prov:repository", "—"),

    
    
            # ✅ Extracted from JSON string
            "Preprocessing Timestamp": preprocessing_info.get("preprocessing_timestamp", "—"),
    
            "Training Start Time": fair4ml.get("fair4ml:trainingStartTime", "—"),
            "Training End Time": fair4ml.get("fair4ml:trainingEndTime", "—"),
    
            "Database Title": fair.get("dc:title", "—"),
            "Database Creator": fair.get("dc:creator", "—"),
            "Database Last Modified": fair.get("dcterms:modified", "—"),
    

        }

  # ✅ Function to extract configuration & evaluation details
    def get_config_and_eval_fields(run_data):
        mlsea = run_data.get("MLSEA", {})
        croissant = run_data.get("Croissant", {})
        fair4ml = run_data.get("FAIR4ML", {})
    
        # Parse hyperparameters and preprocessing_info JSON strings
        try:
            hparams = json.loads(croissant.get("mls:hyperparameters", "{}"))
        except Exception:
            hparams = {}
    
        try:
            prep = json.loads(croissant.get("mls:preprocessingSteps", "{}"))
        except Exception:
            prep = {}
    
        train_test_split = prep.get("train_test_split", {})
    
        return {
            "Target Variable": croissant.get("mls:targetVariable", "—"),
            "Split Strategy": mlsea.get("mlsea:splitStrategy", "Train-Test Split"),
            "Model Architecture": croissant.get("mls:modelArchitecture", "—"),
            "Serialization Format": croissant.get("mls:serializationFormat", "—"),
            "Model Version": croissant.get("mls:modelVersion", "—"),
            "Training Start Time": fair4ml.get("fair4ml:trainingStartTime", "—"),
            "Training End Time": fair4ml.get("fair4ml:trainingEndTime", "—"),
    
            # 📊 Evaluation Metrics - Test
            "Accuracy (Test)": mlsea.get("mlsea:accuracy", "—"),
            "F1 Score (Test)": mlsea.get("mlsea:f1_macro", "—"),
            "Precision (Test)": mlsea.get("mlsea:precision_macro", "—"),
            "Recall (Test)": mlsea.get("mlsea:recall_macro", "—"),
            "ROC AUC (Test)": mlsea.get("mlsea:roc_auc", "—"),
    
            # 📊 Evaluation Metrics - Train
            "Accuracy (Train)": mlsea.get("mlsea:training_accuracy_score", "—"),
            "F1 Score (Train)": mlsea.get("mlsea:training_f1_score", "—"),
            "Precision (Train)": mlsea.get("mlsea:training_precision_score", "—"),
            "Recall (Train)": mlsea.get("mlsea:training_recall_score", "—"),
            "Loss (Train)": mlsea.get("mlsea:training_log_loss", "—"),
    
            # 🧠 Key Hyperparameters
            "Hyperparam → n_estimators": hparams.get("n_estimators", "—"),
            "Hyperparam → max_depth": hparams.get("max_depth", "—"),
            "Hyperparam → min_samples_split": hparams.get("min_samples_split", "—"),
            "Hyperparam → min_samples_leaf": hparams.get("min_samples_leaf", "—"),
            "Hyperparam → criterion": hparams.get("criterion", "—"),
            "Hyperparam → max_features": hparams.get("max_features", "—"),
            "Hyperparam → bootstrap": hparams.get("bootstrap", "—"),
            "Hyperparam → oob_score": hparams.get("oob_score", "—"),
            "Hyperparam → class_weight": "None" if hparams.get("class_weight") is None else hparams.get("class_weight"),
    
            # 🧪 From preprocessing_info
            "Preprocessing → Random State": train_test_split.get("random_state", "—"),
            "Preprocessing → Test Size": train_test_split.get("test_size", "—"),
            "Preprocessing → Timestamp": prep.get("preprocessing_timestamp", "—")
        }


    # ✅ Function to show comparison
    def display_comparison(title, data_fn):
        st.subheader(title)
        data1 = data_fn(run_data_1)

        if compare_mode and run_data_2:
            data2 = data_fn(run_data_2)

            df_display = pd.DataFrame({
                "Field": [str(k) for k in data1.keys()],
                f"Run 1 ({selected_run})": [str(data1.get(k, "—")) for k in data1.keys()],
                f"Run 2 ({second_run})": [str(data2.get(k, "—")) for k in data1.keys()]
            })

            def highlight_diff(row):
                return [
                    "",
                    "background-color: #fbe8e8; color: black" if row[1] != row[2] else "",
                    "background-color: #fbe8e8; color: black" if row[1] != row[2] else ""
                ]

            st.dataframe(df_display.style.apply(highlight_diff, axis=1), use_container_width=True)
        else:
            df_display = pd.DataFrame({
                "Field": [str(k) for k in data1.keys()],
                "Value": [str(v) for v in data1.values()]
            })
            st.dataframe(df_display, use_container_width=True)

    # ✅ Show both sections
    display_comparison("🔍 Provenance & Reproducibility Details", get_provenance_fields)
    display_comparison("🧪 Configuration & Evaluation Strategy", get_config_and_eval_fields)


elif selected == "📣 Notify Outdated Forks":
    st.title("📣 Notify Outdated GitHub Forks")
    st.markdown("""
Automatically detect which collaborators' **forks** of your GitHub repository are **behind** the main branch — and notify them with a GitHub Issue.

🔍 **What it does**:
- Fetches the **latest commit** on your main repository
- Compares it against each fork's latest commit
- Flags forks that are **out-of-date**
- Opens a **GitHub Issue** tagging those collaborators

🔧 **How to use**:
1. Enter your **GitHub username**, **repository name**, and **personal access token**
2. Click **🔔 Notify Fork Owners**
3. A GitHub Issue will be created if any forks are outdated

💡 Ideal for research collaborations, reproducibility checks, and proactive version alignment.
""")

    # Inputs
    owner = st.text_input("GitHub Username", value="reema-dass26")
    repo = st.text_input("Repository Name", value="REPO")
    token = st.text_input("GitHub Personal Access Token", type="password")

    if st.button("🔔 Notify Fork Owners"):
        if not all([owner, repo, token]):
            st.warning("⚠️ Please provide all required inputs.")
        else:
            with st.spinner("🔍 Checking forks against latest commit..."):
                try:
                    headers = {
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }

                    # Step 1: Get latest commit from main repo
                    main_commit_resp = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}/commits",
                        headers=headers,
                        params={"per_page": 1}
                    )
                    main_commit_resp.raise_for_status()
                    latest_sha = main_commit_resp.json()[0]["sha"]
                    st.success(f"✅ Latest main commit: `{latest_sha}`")

                    # Step 2: Get forks
                    forks_resp = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}/forks",
                        headers=headers
                    )
                    forks_resp.raise_for_status()
                    forks = forks_resp.json()

                    # Step 3: Compare commits
                    outdated_forks = []
                    for fork in forks:
                        fork_owner = fork["owner"]["login"]
                        fork_commit_resp = requests.get(
                            fork["url"] + "/commits",
                            headers=headers,
                            params={"per_page": 1}
                        )
                        if fork_commit_resp.status_code != 200:
                            st.warning(f"⚠️ Could not check @{fork_owner}")
                            continue
                        fork_sha = fork_commit_resp.json()[0]["sha"]
                        if fork_sha != latest_sha:
                            outdated_forks.append(fork_owner)

                    # Step 4: Notify
                    if outdated_forks:
                        st.warning(f"🔁 Outdated forks detected: {', '.join(outdated_forks)}")
                        issue_title = "🔔 Fork Sync Needed: Your fork is behind the main repository"
                        issue_body = (
                            f"Hi {' '.join(f'@{user}' for user in outdated_forks)},\n\n"
                            f"The main repository has been updated with commit `{latest_sha}`.\n"
                            "Your fork is currently out of sync. Please pull the latest changes to stay aligned.\n\n"
                            "**Maintainer Notice**"
                        )
                        issue_create_resp = requests.post(
                            f"https://api.github.com/repos/{owner}/{repo}/issues",
                            headers=headers,
                            json={"title": issue_title, "body": issue_body}
                        )
                        if issue_create_resp.status_code == 201:
                            issue_url = issue_create_resp.json().get("html_url")
                            st.success(f"✅ Issue created: [View on GitHub]({issue_url})")
                        else:
                            st.error("❌ Failed to create issue.")
                            st.code(issue_create_resp.text)
                    else:
                        st.success("✅ All forks are up-to-date!")

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

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

    run_ids = df['run_id'].dropna().unique()
    selected_run = st.selectbox("Select a Run (for Justifications)", run_ids)

    run_row = df[df['run_id'] == selected_run]
    if run_row.empty:
        st.warning("⚠️ No metadata found for the selected run.")
    else:
        row_dict = run_row.iloc[0].to_dict()
        justifications = {
            k.replace("MLSEA_justification_", "").replace("_", " ").capitalize(): v
            for k, v in row_dict.items()
            if k.startswith("MLSEA_justification_") and isinstance(v, str) and v.strip()
        }

        if justifications:
            df_just = pd.DataFrame(
                list(justifications.items()),
                columns=["Modeling Decision", "Justification"]
            )
            st.success(f"✅ Loaded justifications for `{selected_run}`")
            st.write("### 📋 Researcher Justification Table")
            st.dataframe(df_just, use_container_width=True)
        else:
            st.info("ℹ️ No justifications were provided in this run.")


# elif selected == "📚 Invenio Metadata":
#     st.title("📚 Invenio Metadata")
#     st.markdown("""
# 📚 View metadata published via **Invenio**, associated with your experiment.

# 🔍 Includes:
# - Title, creators, publication date
# - PID and status info
# - Attached files and view/download counts
# """)

#     provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
#     provenance_folders = [os.path.basename(folder) for folder in provenance_folders]
    
#     if not provenance_folders:
#         st.warning("⚠️ No provenance folders found.")
#     else:
#         selected_run = st.selectbox("Select a Run for Invenio Metadata", provenance_folders)
#         summary_path = os.path.join("MODEL_PROVENANCE", selected_run, f"{selected_run}_run_summary.json")

#         try:
#             with open(summary_path, "r") as f:
#                 run_data = json.load(f)
#             invenio_meta = run_data.get("invenio_metadata", {})

#             if invenio_meta:
#                 df_view = pd.DataFrame([{
#                     "Title": invenio_meta.get("title", ""),
#                     "Creator": invenio_meta.get("creator", ""),
#                     "Published": invenio_meta.get("publication_date", ""),
#                     "Status": invenio_meta.get("status", ""),
#                     "Views": invenio_meta.get("views", 0),
#                     "Downloads": invenio_meta.get("downloads", 0)
#                 }])

#                 st.header("📚 Invenio Metadata Overview")
#                 st.dataframe(df_view, use_container_width=True)

#                 st.header("📁 Files in Publication")
#                 files_list = invenio_meta.get("files", [])
#                 if files_list:
#                     st.json(files_list)
#                 else:
#                     st.info("ℹ️ No files recorded in the publication.")
#             else:
#                 st.warning("ℹ️ No `invenio_metadata` found for this run.")

#         except Exception as e:
#             st.error(f"❌ Error loading Invenio metadata: {e}")

#     import streamlit as st
#     import glob, os

elif selected == "📤 Export Provenance":
    st.title("📤 Export Provenance")

    # 1. Discover available provenance folders
    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]

    if not provenance_folders:
        st.warning("⚠️ No provenance data available.")
    else:
        # 2. Select a run
        selected_run = st.selectbox("Select a Run ID", provenance_folders)
        run_base = os.path.join("MODEL_PROVENANCE", selected_run)

        # 3. Detect available export files
        available_files = os.listdir(run_base)

        json_file = next((f for f in available_files if "export" in f and f.endswith(".json")), None)
        jsonld_file = next((f for f in available_files if f.endswith(".jsonld")), None)
        rdfxml_file = next((f for f in available_files if f.endswith(".rdf")), None)

        # If no exports available
        if not any([json_file, jsonld_file, rdfxml_file]):
            st.warning("⚠️ No exportable provenance files found.")
            st.stop()

        # 4. Let user pick format only from what's available
        format_options = []
        if json_file:
            format_options.append("JSON")
        if jsonld_file:
            format_options.append("JSON-LD")
        if rdfxml_file:
            format_options.append("RDF/XML")

        export_format = st.radio("Choose Export Format", options=format_options)

        # 5. Map to file and MIME
        if export_format == "JSON":
            file_path = os.path.join(run_base, json_file)
            mime = "application/json"
            html_path = None
        elif export_format == "JSON-LD":
            file_path = os.path.join(run_base, jsonld_file)
            mime = "application/ld+json"
            html_path = os.path.join(run_base, "full_provenance_jsonld_viz.html")
        else:
            file_path = os.path.join(run_base, rdfxml_file)
            mime = "application/rdf+xml"
            html_path = os.path.join(run_base, "full_provenance_rdfxml_viz.html")

        # 6. Generate interactive visualization if needed
        if html_path and not os.path.exists(html_path):
            try:
                visualize_interactive_provenance(file_path, html_path)
            except Exception as e:
                st.warning(f"⚠️ Could not generate interactive visualization: {e}")
                html_path = None

        # 7. Offer file download
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label=f"📥 Download {export_format}",
                data=file_bytes,
                file_name=os.path.basename(file_path),
                mime=mime
            )
        else:
            st.error(f"❌ {export_format} file not found for selected run.")
            st.stop()

        # 8. Display HTML visualization if available
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750, scrolling=True)
        elif export_format != "JSON":
            st.info("🔍 No visualization available for this export.")

            
        
        st.title("🔎 Query Your Provenance Data (SPARQL)")
        
        # --- 1. Load provenance file ---
        run_base = os.path.join("MODEL_PROVENANCE", selected_run)
        rdf_files = [f for f in os.listdir(run_base) if f.endswith(".rdf") or f.endswith(".jsonld")]
        
        if not rdf_files:
            st.warning("⚠️ No RDF/JSON-LD provenance files found.")
            st.stop()
        
        selected_file = st.selectbox("Select provenance file", rdf_files)
        full_path = os.path.join(run_base, selected_file)
        rdf_format = "json-ld" if selected_file.endswith(".jsonld") else "xml"
        
        # --- 2. Load RDF graph ---
        g = Graph()
        try:
            g.parse(full_path, format=rdf_format)
            st.success(f"✅ Loaded: {selected_file}")
        except Exception as e:
            st.error(f"❌ Failed to parse RDF file: {e}")
            st.stop()
        
        # --- 3. Preset queries ---
        PRESET_QUERIES = {
            "Show all triples (limit 25)": """
                SELECT ?s ?p ?o WHERE {
                  ?s ?p ?o .
                } LIMIT 25
            """,
            "Who ran the experiment?": """
                SELECT ?agentName WHERE {
                  ?agent a <http://www.w3.org/ns/prov#Agent> ;
                         <http://xmlns.com/foaf/0.1/name> ?agentName .
                }
            """,
            "Which dataset was used in a run?": """
                SELECT ?dataset WHERE {
                  ?run a <http://www.w3.org/ns/prov#Activity> ;
                       <http://www.w3.org/ns/prov#used> ?dataset .
                  FILTER CONTAINS(STR(?dataset), "dataset")
                }
            """,
            "Which model was generated by which run?": """
                SELECT ?model ?run WHERE {
                  ?model a <http://www.w3.org/ns/prov#Entity> ;
                         <http://www.w3.org/ns/prov#wasGeneratedBy> ?run .
                }
            """,
            
            "When did the training start and end?": """
                SELECT ?start ?end WHERE {
                  ?run a <http://www.w3.org/ns/prov#Activity> ;
                       <http://www.w3.org/ns/prov#startedAtTime> ?start ;
                       <http://www.w3.org/ns/prov#endedAtTime> ?end .
                }
            """,
            "List all training accuracy metrics": """
                SELECT ?metric ?value WHERE {
                  ?run a <http://www.w3.org/ns/prov#Activity> ;
                       ?metric ?value .
                  FILTER CONTAINS(STR(?metric), "accuracy")
                }
            """,
            "Get model hyperparameters": """
                SELECT ?param ?value WHERE {
                  ?model a <http://www.w3.org/ns/prov#Entity> ;
                         ?param ?value .
                  FILTER CONTAINS(STR(?param), "hyper")
                }
            """,
           
            "List all entity types in the graph": """
                SELECT DISTINCT ?type WHERE {
                  ?s a ?type .
                }
            """,
        }
        
        # --- 4. Query Mode: Preset or Manual ---
        query_mode = st.radio("Choose Query Mode", ["Use Preset", "Write Your Own"])
        
        if query_mode == "Use Preset":
            preset_key = st.selectbox("Choose a SPARQL query", list(PRESET_QUERIES.keys()))
            sparql_query = st.text_area("SPARQL Query", PRESET_QUERIES[preset_key], height=200)
        else:
            sparql_query = st.text_area("Write your SPARQL query below:", "", height=200)
        
        # --- 5. Execute Query ---
        if st.button("▶️ Run Query"):
            try:
                results = g.query(sparql_query)
                rows = [list(map(str, row)) for row in results]
        
                if not rows:
                    st.info("No results found.")
                else:
                    st.success(f"✅ {len(rows)} results")
                    st.dataframe(rows)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

elif selected == "⚙️ Environment Requirements":
    st.title("⚙️ Environment Requirements")
    st.markdown("""
Download the environment files used during training — including Python packages and versions.

🛠️ These help you:
- Reproduce the experiment
- Set up the same environment elsewhere
""")

    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]

    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        selected_folder = st.selectbox("Select a Run", provenance_folders)
        base_path = os.path.join("MODEL_PROVENANCE", selected_folder)

        # Check for common env files
        req_path = os.path.join(base_path, "requirements.txt")
        env_path = os.path.join(base_path, "environment.yaml")

        if os.path.exists(req_path):
            with open(req_path, "r", encoding="utf-8") as f:
                req_text = f.read()
            st.download_button("⬇️ Download requirements.txt", req_text, file_name="requirements.txt")

        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_text = f.read()
            st.download_button("⬇️ Download environment.yaml", env_text, file_name="environment.yaml")

        if not os.path.exists(req_path) and not os.path.exists(env_path):
            st.info("ℹ️ No environment files (`requirements.txt` or `environment.yaml`) found for this run.")
