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

from rdflib import Graph, Namespace
from pyvis.network import Network

def visualize_interactive_provenance(rdf_file, output_html="provenance_graph.html", max_edges=150):
    g = Graph()
    g.parse(rdf_file)

    # Namespace for type checking
    PROV = Namespace("http://www.w3.org/ns/prov#")

    # Build type map first
    node_types = {}
    for s, p, o in g.triples((None, RDF.type, None)):
        node_types[str(s)] = str(o)

    # Initialize PyVis
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.force_atlas_2based()

    seen_nodes = set()
    edge_count = 0

    for s, p, o in g:
        if edge_count >= max_edges:
            break

        s_label = str(s)
        o_label = str(o)
        p_label = str(p).split("#")[-1] if "#" in str(p) else str(p).split("/")[-1]

        # Skip long literals
        if len(o_label) > 200:
            o_label = o_label[:200] + "..."

        if s_label.startswith("_:"):
            s_label = f"BlankNode:{s_label[-5:]}"
        if o_label.startswith("_:"):
            o_label = f"BlankNode:{o_label[-5:]}"

        def color_for(node):
            if node_types.get(node) == str(PROV.Agent):
                return "gold"
            elif node_types.get(node) == str(PROV.Activity):
                return "tomato"
            elif node_types.get(node) == str(PROV.Entity):
                return "dodgerblue"
            else:
                return "lightgray"

        # Add subject node
        if s_label not in seen_nodes:
            net.add_node(s_label, label=s_label[:40], title=s_label, color=color_for(s_label), font={'size': 14})
            seen_nodes.add(s_label)

        # Add object node
        if o_label not in seen_nodes:
            net.add_node(o_label, label=o_label[:40], title=o_label, color=color_for(o_label), font={'size': 14})
            seen_nodes.add(o_label)

        # Add edge
        net.add_edge(s_label, o_label, label=p_label)
        edge_count += 1

    net.show(output_html)
    return output_html

import os
import json
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, DCTERMS, FOAF, XSD


def generate_grouped_metadata_json(run_summary_path, output_grouped_path):
    with open(run_summary_path, "r") as f:
        run_data = json.load(f)

    relevant_fields = {
        "RQ1.1_Data_Provenance": [
            "Internal_DBRepo_feature_names",
            "Internal_DBRepo_dropped_columns",
            "Internal_DBRepo_n_records",
            "FAIR_dataset_title",
            "FAIR_dataset_identifier",
            "FAIR_dataset_creator",
            "FAIR_dataset_license",
            "FAIR_dataset_access_url",
            "FAIR_dataset_documentation",
            "FAIR_dataset_keywords",
            "FAIR_dataset_publication_date",
            "FAIR_dataset_publisher",
            "MLSEA_dataPreprocessing"
        ],
        "RQ1.2_Model_Provenance": [
            "MLSEA_hyperparameters",
            "MLSEA_modelArchitecture",
            "MLSEA_trainingProcedure",
            "MLSEA_trainingCodeSnapshot",
            "MLSEA_evaluationMetrics",
            "ML_EXP_params",
            "ML_EXP_metrics",
            "mlflow.log-model.history",
            "ML_EXP_dataset_name",
            "ML_EXP_dataset_version",
            "ML_EXP_model_name",
            "ML_EXP_notebook_name"
        ],
        "RQ2_Metadata_Audit_Tracing": [
            "GIT_code_version",
            "GIT_current_commit_hash",
            "GIT_user",
            "GIT_user_email",
            "MLSEA_modelPath",
            "Internal_DBRepo_target_name",
            "MLSEA_performanceInterpretation",
            "ML_EXP_tags",
            "ML_EXP_artifacts"
        ],
        "RQ4_Schema_Mapping_Interoperability": [
            "PROV-O_prov_Activity",
            "PROV-O_prov_used",
            "PROV-O_prov_Entity",
            "PROV-O_prov_location",
            "PROV-O_prov_wasAssociatedWith",
            "PROV-O_prov_wasGeneratedBy",
            "FAIR4ML_target_variable",
            "FAIR4ML_ml_task",
            "FAIR4ML_serializationFormat",
            "FAIR4ML_dataset_dataset_type",
            "FAIR4ML_hasCO2eEmissions"
        ]
    }

    grouped_output = {}
    for section, fields in relevant_fields.items():
        grouped_output[section] = {
            field: run_data.get(field) or run_data.get("ML_EXP_tags", {}).get(field, "Not available")
            for field in fields
        }

    with open(output_grouped_path, "w") as out:
        json.dump(grouped_output, out, indent=2)

    return output_grouped_path


def export_full_provenance_rdf(grouped_metadata_path, output_basename="full_provenance"):
    with open(grouped_metadata_path, "r") as f:
        grouped = json.load(f)

    g = Graph()
    PROV = Namespace("http://www.w3.org/ns/prov#")
    SCHEMA = Namespace("http://schema.org/")
    MLS = Namespace("http://www.w3.org/ns/mls#")
    EX = Namespace("http://example.org/")
    g.bind("prov", PROV)
    g.bind("schema", SCHEMA)
    g.bind("mls", MLS)
    g.bind("dcterms", DCTERMS)
    g.bind("foaf", FOAF)
    g.bind("ex", EX)

    rq1 = grouped["RQ1.1_Data_Provenance"]
    rq2 = grouped["RQ2_Metadata_Audit_Tracing"]
    rq3 = grouped["RQ1.2_Model_Provenance"]
    rq4 = grouped["RQ4_Schema_Mapping_Interoperability"]

    model_name = rq3.get("ML_EXP_model_name", "unknown_model")
    dataset_uri = URIRef(EX[f"{model_name}_dataset"])
    activity_uri = URIRef(EX[f"{model_name}_training"])
    agent_uri = URIRef(EX["Reema_George_Dass"])

    def safe_literal(value):
        if value and value != "Not available":
            return Literal(value)
        return None

    def add_dict_as_nodes(parent_uri, predicate, data_dict):
        for k, v in data_dict.items():
            if v and v != "Not available":
                node = BNode()
                g.add((parent_uri, predicate, node))
                g.add((node, SCHEMA.name, Literal(k)))
                g.add((node, SCHEMA.value, Literal(str(v))))

    g.add((dataset_uri, RDF.type, PROV.Entity))
    for field in [
        ("FAIR_dataset_title", DCTERMS.title),
        ("FAIR_dataset_identifier", DCTERMS.identifier),
        ("FAIR_dataset_creator", DCTERMS.creator),
        ("FAIR_dataset_license", DCTERMS.license),
        ("FAIR_dataset_documentation", DCTERMS.description),
        ("FAIR_dataset_access_url", SCHEMA.url),
        ("FAIR_dataset_keywords", SCHEMA.keywords),
        ("FAIR_dataset_publication_date", DCTERMS.issued),
        ("FAIR_dataset_publisher", DCTERMS.publisher),
    ]:
        val = safe_literal(rq1.get(field[0]))
        if val:
            g.add((dataset_uri, field[1], val))
    g.add((dataset_uri, PROV.wasGeneratedBy, activity_uri))
    g.add((dataset_uri, PROV.wasAttributedTo, agent_uri))

    g.add((agent_uri, RDF.type, PROV.Agent))
    g.add((agent_uri, FOAF.name, safe_literal(rq2.get("GIT_user", "Unknown"))))
    g.add((agent_uri, FOAF.mbox, safe_literal(rq2.get("GIT_user_email", ""))))

    g.add((activity_uri, RDF.type, PROV.Activity))
    g.add((activity_uri, PROV.wasAssociatedWith, agent_uri))
    if rq4.get("PROV-O_prov_used"):
        g.add((activity_uri, PROV.used, URIRef(rq4["PROV-O_prov_used"])))
    if rq4.get("FAIR4ML_ml_task"):
        g.add((activity_uri, MLS.taskType, Literal(rq4["FAIR4ML_ml_task"])))
    if rq4.get("PROV-O_prov_startedAtTime") and rq4["PROV-O_prov_startedAtTime"] != "info not available":
        g.add((activity_uri, PROV.startedAtTime, Literal(rq4["PROV-O_prov_startedAtTime"], datatype=XSD.dateTime)))
    if rq4.get("PROV-O_prov_endedAtTime"):
        g.add((activity_uri, PROV.endedAtTime, Literal(rq4["PROV-O_prov_endedAtTime"], datatype=XSD.dateTime)))
    if rq4.get("PROV-O_prov_location"):
        g.add((activity_uri, PROV.atLocation, URIRef(rq4["PROV-O_prov_location"])))
    if rq2.get("GIT_current_commit_hash"):
        g.add((activity_uri, PROV.value, Literal(f"Git commit: {rq2['GIT_current_commit_hash']}")))

    for field, pred in [
        ("MLSEA_modelArchitecture", MLS.modelArchitecture),
        ("MLSEA_trainingProcedure", MLS.trainingProcedure),
        ("MLSEA_trainingCodeSnapshot", SCHEMA.codeRepository)
    ]:
        val = safe_literal(rq3.get(field))
        if val:
            g.add((activity_uri, pred, val))

    try:
        metrics = rq3.get("ML_EXP_metrics", {})
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        add_dict_as_nodes(activity_uri, MLS.hasEvaluationMeasure, metrics)
    except Exception:
        pass

    try:
        params = rq3.get("ML_EXP_params", {})
        if isinstance(params, str):
            params = json.loads(params)
        add_dict_as_nodes(activity_uri, MLS.hasHyperParameter, params)
    except Exception:
        pass

    try:
        preprocessing = rq1.get("MLSEA_dataPreprocessing", {})
        if isinstance(preprocessing, str):
            preprocessing = json.loads(preprocessing)
        add_dict_as_nodes(activity_uri, MLS.dataPreparation, preprocessing)
    except Exception:
        pass

    for k, v in rq2.get("ML_EXP_tags", {}).items():
        if k.startswith("justification_") or k.startswith("MLSEA_justification"):
            node = BNode()
            g.add((activity_uri, PROV.wasInfluencedBy, node))
            g.add((node, SCHEMA.name, Literal(k)))
            g.add((node, SCHEMA.description, Literal(v)))

    if "MLSEA_improvedFrom" in rq2.get("ML_EXP_tags", {}):
        previous = rq2["ML_EXP_tags"]["MLSEA_improvedFrom"]
        if previous and previous != "None":
            g.add((activity_uri, PROV.wasDerivedFrom, URIRef(EX[previous])))

    jsonld_path = f"{output_basename}.jsonld"
    rdfxml_path = f"{output_basename}.rdf"
    g.serialize(destination=jsonld_path, format="json-ld", indent=2)
    g.serialize(destination=rdfxml_path, format="xml")

    return jsonld_path, rdfxml_path


import os
import glob
import json
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """
    Load and fully flatten JSON metadata files from the MODEL_PROVENANCE directory.
    Returns a DataFrame with one row per run summary.
    """
    pattern = os.path.join("MODEL_PROVENANCE", "*", "*_run_summary.json")
    files = glob.glob(pattern)

    if not files:
        st.warning("⚠️ No run summary JSON files found inside MODEL_PROVENANCE!")
        return pd.DataFrame()

    rows = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)

            row = {}

            # Flatten top-level keys (skip nested dicts)
            for k, v in summary.items():
                if not isinstance(v, dict):
                    row[k] = v

            # Flatten nested sections with prefixes
            for section_key in ["params", "metrics", "tags"]:
                section = summary.get(section_key, {})
                for k, v in section.items():
                    row[f"{section_key}_{k}"] = v

            rows.append(row)
            print(f"✅ Parsed: {row.get('run_id', 'UNKNOWN')} from {file_path}")

        except Exception as e:
            st.error(f"❌ Error loading file {file_path}: {e}")

    if not rows:
        st.warning("⚠️ No valid run summary data could be loaded!")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    st.success(f"✅ Loaded {len(df)} runs with {len(df.columns)} columns.")
    print(df.columns)
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

from rdflib import Graph
import networkx as nx
import matplotlib.pyplot as plt

# def generate_viz_if_missing(rdf_path, output_img_path):
#     if not os.path.exists(rdf_path):
#         return None
#     if not os.path.exists(output_img_path):
#         # Load RDF
#         g = Graph()
#         g.parse(rdf_path)
#         # Build graph
#         G = nx.DiGraph()
#         for s, p, o in g:
#             G.add_edge(str(s), str(o), label=str(p))
#         pos = nx.spring_layout(G, k=0.6)
#         plt.figure(figsize=(16, 12))
#         nx.draw(G, pos, with_labels=True, node_size=2000, font_size=8, node_color="lightblue", edge_color="gray", arrows=True)
#         edge_labels = nx.get_edge_attributes(G, 'label')
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
#         plt.title("Provenance Graph")
#         plt.tight_layout()
#         plt.savefig(output_img_path)
#         plt.close()

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
            "📚 Invenio Metadata",
            "📦 Environment Requirements"

            
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
        {"emoji": "📚", "title": "Invenio Metadata", "desc": "Render Invenio-style metadata records for datasets and publications."}
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
        st.warning("⚠️ No runs found. Please train a model first.")
    else:
        selected_run = st.selectbox("Select a Run ID", run_ids)
        run_df = df[df["run_id"] == selected_run]

        if run_df.empty:
            st.warning("No metadata available for this run.")
        else:
            flat_row = run_df.iloc[0].to_dict()

            # ==== PROV-O Metadata ====
            # ==== PROV-O Metadata ====
            st.subheader("🛰️ Dataset Provenance (PROV-O)")
            prov_fields = {
    "Entity": flat_row.get("tags_DOI_prov_entity", "—"),
    "Activity": flat_row.get("tags_DOI_prov_activity", "—"),
    "Agent (Dataset Creator)": flat_row.get("tags_DOI_prov_agent_dataset_creator", "—"),
    "Agent (Database Creator)": flat_row.get("tags_DOI_prov_role_database_creator", "—"),
    "Used Source": flat_row.get("tags_DOI_prov_used", "—"),
    "Started At": flat_row.get("tags_DOI_prov_startedAtTime", "—"),
    "Was Attributed To": flat_row.get("tags_DOI_prov_wasAttributedTo", "—"),
    "Was Derived From": flat_row.get("tags_DOI_prov_wasDerivedFrom", "—"),
    "Was Generated By": flat_row.get("tags_DOI_prov_wasGeneratedBy", flat_row.get("tags_DOI_prov_role_database_creator", "—"))
}

                
            st.dataframe(pd.DataFrame(list(prov_fields.items()), columns=["Field", "Value"]), use_container_width=True)


            # ==== FAIR / DOI Dataset Metadata ====
            st.subheader("📚 Dataset Metadata (DOI-based)")
            fair_fields = {
    "Title": flat_row.get("tags_DOI_dataset_title", "—"),
    "Creator": flat_row.get("tags_DOI_dataset_creator", "—"),
    "Publisher": flat_row.get("tags_DOI_dataset_publisher", "—"),
    "Publication Year": flat_row.get("tags_DOI_dataset_publication_date", "—"),
    "DOI": flat_row.get("tags_DOI_dataset_id", "—"),
    "Access URL": flat_row.get("tags_DOI_dataset_access_url", "—"),
    "Documentation": flat_row.get("tags_DOI_dataset_documentation", "—"),
    "License": flat_row.get("tags_DOI_dataset_license", "—"),
    "Metadata Standard": flat_row.get("tags_DOI_metadata_standard", "—"),
    "Keywords": flat_row.get("tags_DOI_dataset_keywords", "—"),
    "Related Resources": flat_row.get("tags_DOI_related_resources", "—")
}


            st.dataframe(pd.DataFrame(list(fair_fields.items()), columns=["Field", "Value"]), use_container_width=True)

            # ==== Internal DBRepo Metadata ====
            st.subheader("🏛️ Internal DBRepo Metadata")
            db_fields = {
    "Dataset ID": flat_row.get("tags_dataset_id", "—"),
    "Dataset Name": flat_row.get("tags_dataset_name", "—"),
    "Version": flat_row.get("tags_dataset_version", "—"),
    "Description": flat_row.get("tags_DOI_dataset_description", "—"),  # fill from DOI
    "ML Task": flat_row.get("tags_ml_task", "classification"),  # hardcoded since it's known
    "Target Variable": flat_row.get("tags_target_variable", "—"),
    "Number of Records": flat_row.get("params_input_row_count") or flat_row.get("metrics_row_count_start", "—"),
    "Access URL": flat_row.get("tags_data_source", "—"),
    "Last Modified": flat_row.get("tags_Internal_DBRepo_table_last_modified", "—")
}

            st.dataframe(pd.DataFrame(list(db_fields.items()), columns=["Field", "Value"]), use_container_width=True)

            # ==== Preprocessing Metadata ====
            # ==== Preprocessing Metadata ====
            # ==== Preprocessing Metadata ====
            st.subheader("🧪 Preprocessing & Transformations")
            
            try:
                # Handle both string and already-parsed dicts
                prep_raw = flat_row.get("tags_preprocessing_info") or flat_row.get("MLSEA_dataPreprocessing")
            
                if isinstance(prep_raw, str):
                    prep_dict = json.loads(prep_raw)
                elif isinstance(prep_raw, dict):
                    prep_dict = prep_raw
                else:
                    prep_dict = {}
            
                if prep_dict:
                    # Build a clean display table from selected fields
                    preprocessing_fields = {
                        "Dropped Columns": prep_dict.get("dropped_columns", "—"),
                        "Numeric Columns": prep_dict.get("numeric_columns", "—"),
                        "Target Column": prep_dict.get("target_column", "—"),
                        "Encoding Strategy": prep_dict.get("encoding_strategy", "—"),
                        "Feature Selection": prep_dict.get("feature_selection", "—"),
                        "Coercion Strategy": prep_dict.get("coercion_strategy", "—"),
                        "Missing Value Strategy": prep_dict.get("missing_value_strategy", "—"),
                        "Outlier Detection": prep_dict.get("outlier_detection", "—"),
                        "Scaling": prep_dict.get("scaling", "—"),
                        "Sampling": prep_dict.get("sampling", "—"),
                        "Stratified Split": prep_dict.get("stratified", "—"),
                        "Test Size": prep_dict.get("train_test_split", {}).get("test_size", "—"),
                        "Random State": prep_dict.get("train_test_split", {}).get("random_state", "—"),
                        "Imbalance Ratio": prep_dict.get("imbalance_ratio", "—"),
                        "Timestamp": prep_dict.get("preprocessing_timestamp", "—")
                    }
            
                    st.dataframe(pd.DataFrame(preprocessing_fields.items(), columns=["Field", "Value"]), use_container_width=True)
                else:
                    st.info("ℹ️ No preprocessing metadata found.")
            
            except Exception as e:
                st.warning(f"⚠️ Could not parse preprocessing JSON: {e}")



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
    commit_field = "tags_git_commit"
    version_field = "tags_GIT_code_version"

    version_map = {}

    if commit_field in df.columns:
        if version_field not in df.columns:
            df[version_field] = "untagged"

        version_map = df.set_index(commit_field)[version_field].to_dict()

        st.markdown("### 🏷️ Git Commit – Version Mapping")
        if current_hash:
            st.markdown(f"### 📌 Current Git Commit: `{current_hash}`")
        st.dataframe(df[["run_id", commit_field, version_field]], use_container_width=True)
    else:
        st.warning(f"⚠️ Column `{commit_field}` not found in metadata.")

    deprecated_versions_input = st.text_area("Enter deprecated version tags (one per line):", height=100)
    simulate_current = st.checkbox("☢️ Also mark current local commit as deprecated")
    current_version_tag = version_map.get(current_hash, "untagged")

    if simulate_current:
        deprecated_versions_input += f"\n{current_version_tag}"
        st.info(f"☢️ Added current version `{current_version_tag}` to deprecated list.")

    deprecated_versions = [v.strip() for v in deprecated_versions_input.splitlines() if v.strip()]

    def detect_deprecated_versions(df, deprecated_versions):
        if version_field not in df.columns:
            return pd.DataFrame()
        affected = df[df[version_field].isin(deprecated_versions)].copy()
        if "tags_GIT_user_email" in df.columns:
            affected["github_user"] = df["tags_GIT_user_email"].str.extract(r"\+([a-zA-Z0-9\-]+)@users")[0]
        return affected

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
                        impacted_users = st.session_state.results_df["github_user"].dropna().unique()
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


elif selected == "🧠 ML Model Metadata":
    st.title("🧠 ML Model Metadata")
    st.markdown("""
Explore structured ML model metadata from each experiment.

🔍 What’s covered:
- Hyperparameters, metrics, and justifications  
- Compute environment and training timeline  
- Git + model provenance  
""")

    run_ids = df['run_id'].dropna().unique()
    if not run_ids.any():
        st.warning("⚠️ No runs found. Please train a model first.")
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

        # 🚀 Overview
        show_section("🚀 Model Overview", {
    "Model Name": row.get("tags_model_name", "—"),
    "Architecture": row.get("tags_model_architecture", "—"),
    "Notebook": row.get("tags_notebook_name", "—"),
    "Run Name": row.get("tags_mlflow.runName", "—"),
    "Experiment ID": row.get("experiment_id", "—")
})


        # 🧠 Hyperparameters
        try:
            hyper = json.loads(row.get("tags_hyperparameters", "{}"))
        except Exception:
            hyper = {}
        show_section("🧠 Hyperparameters", hyper)


        # 📊 Evaluation Metrics
        metric_fields = [
    ("Accuracy", "metrics_accuracy"),
    ("F1 Score", "metrics_f1_macro"),
    ("Precision", "metrics_precision_macro"),
    ("Recall", "metrics_recall_macro"),
    ("ROC AUC", "metrics_roc_auc")
]
        metrics = {name: row.get(col, "—") for name, col in metric_fields}
        show_section("📊 Evaluation Metrics", metrics)


        # 🧰 Compute Environment
        try:
            compute_env = json.loads(row.get("tags_compute_environment", "{}"))
        except Exception:
            compute_env = {}
        show_section("🧰 Compute Environment", compute_env)

        # 🕓 Training Timeline
        show_section("🕓 Training Timeline", {
            "Training Start Time": row.get("tags_training_start_time", "—"),
            "Training End Time": row.get("tags_training_end_time", "—"),
            "Test Size": row.get("param_test_size", row.get("tags_test_size", "—"))      ,
            "Random Seed": row.get("params_random_seed", row.get("param_random_state", "—")),
            "Target Variable": row.get("tags_target_variable", "—")
        })


        # 📋 Justifications
        justifications = {
            k.replace("tags_justification_", "").replace("_", " ").capitalize(): v
            for k, v in row.items()
            if k.startswith("tags_justification_") and isinstance(v, str) and v.strip()
        }
        if justifications:
            show_section("📋 Researcher Justifications", justifications)
        else:
            st.info("ℹ️ No justifications recorded for this run.")


        # 🔗 Git Provenance
        show_section("🔗 Git & Versioning", {
            "Git Commit Hash": row.get("tags_git_commit", "—"),
            "Code Version": row.get("tags_GIT_code_version", "—"),
            "Author": row.get("tags_GIT_user", "—"),
            "Email": row.get("tags_GIT_user_email", "—"),
            "Commit URL": row.get("tags_git_commit", "—")
        })



# elif selected == "📊 Model Plots":
#     st.title("📊 Model Explainability & Evaluation Plots")
#     st.markdown("""
# Visualize how your machine learning model is performing — and understand **why** it's making the predictions it does.

# 🔗 This section links each plot back to the run ID, dataset, and model used to generate it.
# """)

#     import glob
#     import json
#     import os

#     # Step 1: Build mapping of folder names to run_ids based on summary files

#     folder_paths = glob.glob(os.path.join("MODEL_PROVENANCE", "*_run_summary.json")) + \
#                glob.glob(os.path.join("MODEL_PROVENANCE", "*", "*_run_summary.json"))

    
    
#     run_id_to_folder = {}

#     for path in folder_paths:
#         folder = os.path.dirname(path)
#         folder_name = os.path.basename(folder)
#         run_id = folder_name  # Assuming folder name is equal to run_id
#         run_id_to_folder[run_id] = folder

#     # Step 2: Filter df to only those with matching folders
#     valid_run_ids = df["run_id"].dropna().unique()
#     valid_run_ids = [r for r in valid_run_ids if r in run_id_to_folder]

#     if not valid_run_ids:
#         st.warning("No valid run folders found that match run IDs in metadata.")
#         st.stop()

#     # Step 3: Let user select a valid run_id
#     selected_run = st.selectbox("Select a Run ID", sorted(valid_run_ids))
#     run_df = df[df["run_id"] == selected_run]

#     if run_df.empty:
#         st.error(f"No metadata found for selected run ID: {selected_run}")
#         st.stop()

#     run_data = run_df.iloc[0].to_dict()
#     run_folder = run_id_to_folder[selected_run]

#     st.success(f"📁 Loaded metadata from: `{selected_run}` at `{run_folder}`")

#     # ── Extended Metadata ──
#     with st.expander("📋 Extended Metadata"):

#         def safe_str(val):
#             if isinstance(val, (dict, list)):
#                 return json.dumps(val)
#             elif val is None:
#                 return "—"
#             return str(val)

#         meta_preview = {
#             "Run ID": run_data.get("run_id", "—"),
#             "Model Name": run_data.get("tags_model_name", "—"),
#             "Dataset Title": run_data.get("tags_DOI_dataset_title", "—"),
#             "Training Start": run_data.get("tags_training_start_time", "—"),
#             "Training End": run_data.get("tags_training_end_time", "—"),
#             "Accuracy (Test)": run_data.get("metrics_accuracy", "—"),
#             "F1 Macro (Test)": run_data.get("metrics_f1_macro", "—"),
#             "Precision (Test)": run_data.get("metrics_precision_macro", "—"),
#             "Recall (Test)": run_data.get("metrics_recall_macro", "—"),
#             "ROC AUC (Test)": run_data.get("metrics_roc_auc", "—"),
#             "Training Accuracy": run_data.get("metrics_training_accuracy_score", "—"),
#             "Target Variable": run_data.get("tags_target_variable", "—"),
#             "Serialization Format": run_data.get("tags_model_serialization", "—"),
#             "Model Path": run_data.get("tags_model_path", "—"),
#             "Improved From": run_data.get("tags_MLSEA_improvedFrom", "—"),
#             "Training Code Snapshot": run_data.get("tags_justification_training_code_snapshot", "—"),
#             "Training Procedure": run_data.get("tags_justification_training_procedure", "—")
#         }

#         try:
#             hparams = json.loads(run_data.get("tags_hyperparameters", "{}"))
#         except:
#             hparams = {}

#         for k, v in hparams.items():
#             meta_preview[f"Hyperparam → {k}"] = v

#         try:
#             prep = json.loads(run_data.get("tags_preprocessing_info", "{}"))
#         except:
#             prep = {}

#         for k in ["dropped_columns", "final_feature_columns", "target_column"]:
#             if k in prep:
#                 meta_preview[f"Preprocessing → {k}"] = prep[k]

#         cleaned = {k: safe_str(v) for k, v in meta_preview.items()}
#         st.dataframe(pd.DataFrame(list(cleaned.items()), columns=["Field", "Value"]), use_container_width=True)

#     # ── Plot Viewer ──
#     st.markdown("### 📈 Select and View Plot")

#     plot_files = glob.glob(os.path.join(run_folder, "*.png"))

#     if not plot_files:
#         st.warning("⚠️ No plots found in the run folder.")
#         st.stop()

#     plot_options = {}
#     for fpath in plot_files:
#         fname = os.path.basename(fpath).replace(".png", "")
#         label = fname.replace("_", " ").title()
#         plot_options[label] = fpath

#     selected_plot = st.selectbox("Choose Plot", list(plot_options.keys()))
#     plot_path = plot_options[selected_plot]

#     plot_width = st.slider("Adjust Plot Width", 400, 1000, 600)
#     st.image(plot_path, caption=f"{selected_plot} — {selected_run}", width=plot_width)

#     # ── Interpretation ──
#     explanations = {
#         "Feature Importances": "Shows which features contribute most to predictions.",
#         "Shap Summary": "SHAP values show feature impact and distribution.",
#         "Roc Curve": "Visualizes true vs. false positive rates.",
#         "Precision Recall": "Helps evaluate classifier performance under class imbalance.",
#         "Confusion Matrix": "Compares predicted vs. actual outcomes."
#     }

#     for key, explanation in explanations.items():
#         if key.lower() in selected_plot.lower():
#             st.markdown(f"**Interpretation:** {explanation}")
#             break
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
    summary_path = glob.glob(os.path.join(run_folder, "*_run_summary.json"))
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
            "Model Name": run_data.get("tags_model_name", "—"),
            "Dataset Title": run_data.get("tags_DOI_dataset_title", "—"),
            "Training Start": run_data.get("tags_training_start_time", "—"),
            "Training End": run_data.get("tags_training_end_time", "—"),
            "Accuracy (Test)": run_data.get("metrics_accuracy", "—"),
            "F1 Macro (Test)": run_data.get("metrics_f1_macro", "—"),
            "Precision (Test)": run_data.get("metrics_precision_macro", "—"),
            "Recall (Test)": run_data.get("metrics_recall_macro", "—"),
            "ROC AUC (Test)": run_data.get("metrics_roc_auc", "—"),
            "Training Accuracy": run_data.get("metrics_training_accuracy_score", "—"),
            "Target Variable": run_data.get("tags_target_variable", "—"),
            "Serialization Format": run_data.get("tags_model_serialization", "—"),
            "Model Path": run_data.get("tags_model_path", "—"),
            "Improved From": run_data.get("tags_MLSEA_improvedFrom", "—"),
            "Training Code Snapshot": run_data.get("tags_justification_training_code_snapshot", "—"),
            "Training Procedure": run_data.get("tags_justification_training_procedure", "—")
        }

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
        for _, row in df.iterrows():
            mapping_records.append({
                "Run ID": row.get("run_id", "—"),
                "Model Name": row.get("tags_model_name", "—"),
                "Architecture": row.get("tags_model_architecture", "—"),
                "Dataset Title": row.get("tags_DOI_dataset_title", "—"),
                "Dataset Version": row.get("tags_dataset_version", "—"),
                "Dataset Access URL": row.get("tags_data_source", "—"),
                "Accuracy (Test)": row.get("metrics_accuracy", "—"),
                "F1 Score (Test)": row.get("metrics_f1_macro", "—"),
                "ROC AUC (Test)": row.get("metrics_roc_auc", "—")
            })

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

    # ✅ Function to extract provenance fields
    def get_provenance_fields(run_data):
        try:
            preprocessing = json.loads(run_data.get("tags_preprocessing_info", "{}"))
        except:
            preprocessing = {}

        return {
            "Run ID": run_data.get("run_id", "—"),
            "Dataset Title": run_data.get("tags_DOI_dataset_title", "—"),
            "Dataset Version": run_data.get("tags_dataset_version", "—"),
            "Dataset Source URL": run_data.get("tags_data_source", "—"),
            "Notebook Name": run_data.get("tags_notebook_name", "—"),
            "Model Path": run_data.get("tags_model_path", "—"),
            "Model Architecture": run_data.get("tags_model_architecture", "—"),
            "Training Code Snapshot": run_data.get("tags_justification_training_code_snapshot", "—"),
            "Git Commit Hash": run_data.get("tags_git_commit", "—"),
            "Git Commit URL": run_data.get("tags_GIT_current_commit_url", "—"),#TODO
            "Preprocessing Hash": run_data.get("tags_preprocessing_hash", "—"),
            "Preprocessing Timestamp": preprocessing.get("preprocessing_timestamp", "—"),
            "Training Start Time": run_data.get("tags_training_start_time", "—"),
            "Training End Time": run_data.get("tags_training_end_time", "—"),
            "Database Title": run_data.get("tags_dataset_name", "—"),
            "Database Creator": run_data.get("tags_DOI_dataset_creator", "—"),
            "Database Last Modified": run_data.get("tags_DOI_prov_wasGeneratedBy", "—"),
            "Generated By": run_data.get("tags_DOI_prov_wasGeneratedBy", "—"),
            "Used Source": run_data.get("tags_DOI_prov_used", "—"),
            "Activity": run_data.get("tags_DOI_prov_activity", "—")
        }

    # ✅ Function to extract configuration & evaluation details
    def get_config_and_eval_fields(run_data):
        return {
            "Target Variable": run_data.get("tags_target_variable", "—"),
            "Split Strategy": run_data.get("tags_justification_test_split", "—"),
            "Model Architecture": run_data.get("tags_model_architecture", "—"),
            "Accuracy (Test)": run_data.get("metrics_accuracy", "—"),
            "F1 Score (Test)": run_data.get("metrics_f1_macro", "—"),
            "Precision (Test)": run_data.get("metrics_precision_macro", "—"),
            "Recall (Test)": run_data.get("metrics_recall_macro", "—"),
            "ROC AUC (Test)": run_data.get("metrics_roc_auc", "—"),
            "Accuracy (Train)": run_data.get("metrics_training_accuracy_score", "—"),
            "F1 Score (Train)": run_data.get("metrics_training_f1_score", "—"),
            "Precision (Train)": run_data.get("metrics_training_precision_score", "—"),
            "Recall (Train)": run_data.get("metrics_training_recall_score", "—"),
            "Loss (Train)": run_data.get("metrics_training_log_loss", "—"),
            "Hyperparam → n_estimators": run_data.get("params_n_estimators", "—"),
            "Hyperparam → max_depth": run_data.get("params_max_depth", "—"),
            "Hyperparam → min_samples_split": run_data.get("params_min_samples_split", "—"),
            "Hyperparam → min_samples_leaf": run_data.get("params_min_samples_leaf", "—"),
            "Hyperparam → criterion": run_data.get("params_criterion", "—"),
            "Hyperparam → max_features": run_data.get("params_max_features", "—"),
            "Hyperparam → bootstrap": run_data.get("params_bootstrap", "—"),
            "Hyperparam → oob_score": run_data.get("params_oob_score", "—"),
            "Hyperparam → class_weight": run_data.get("params_class_weight", "—"),
            "Hyperparam → random_state": run_data.get("tags_random_state", "—")
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

    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]

    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        selected_folder = st.selectbox("Select a Run (for Justifications)", provenance_folders)
        summary_path = os.path.join("MODEL_PROVENANCE", selected_folder, f"{selected_folder}_run_summary.json")

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            justifications = {
                k.replace("justification_", "").replace("_", " ").capitalize(): v
                for k, v in summary.get("tags", {}).items()
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
📚 View metadata published via **Invenio**, associated with your experiment.

🔍 Includes:
- Title, creators, publication date
- PID and status info
- Attached files and view/download counts
""")

    provenance_folders = glob.glob(os.path.join("MODEL_PROVENANCE", "RandomForest_Iris_v*"))
    provenance_folders = [os.path.basename(folder) for folder in provenance_folders]
    
    if not provenance_folders:
        st.warning("⚠️ No provenance folders found.")
    else:
        selected_run = st.selectbox("Select a Run for Invenio Metadata", provenance_folders)
        summary_path = os.path.join("MODEL_PROVENANCE", selected_run, f"{selected_run}_run_summary.json")

        try:
            with open(summary_path, "r") as f:
                run_data = json.load(f)
            invenio_meta = run_data.get("invenio_metadata", {})

            if invenio_meta:
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

    import streamlit as st
    import glob, os

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
            "Which code version was used in training?": """
                SELECT ?codeHash WHERE {
                  ?run a <http://www.w3.org/ns/prov#Activity> ;
                       <http://www.w3.org/ns/prov#used> ?code .
                  ?code <http://example.org/commit_hash> ?codeHash .
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
            "Which model used a specific dataset?": """
               SELECT ?model ?datasetLabel WHERE {
  ?run a <http://www.w3.org/ns/prov#Activity> ;
       <http://www.w3.org/ns/prov#used> ?dataset .
  ?model <http://www.w3.org/ns/prov#wasGeneratedBy> ?run .
  ?dataset <http://example.org/title> ?datasetLabel .
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