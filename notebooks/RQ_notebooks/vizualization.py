import streamlit as st
import os
import pandas as pd
import json
import plotly.graph_objects as go
import ast
import glob
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
## --------------------------------------------
# Helper Functions
# --------------------------------------------
from typing import List, Dict, Any
from pprint import pprint
from streamlit_option_menu import option_menu
from pyvis.network import Network
import streamlit.components.v1 as components
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="ML Provenance Dashboard", layout="wide")
# Right-corner floating box with vibrant agraph
# with st.container():

#     st.markdown("""
# <div style='background: linear-gradient(to right, #ff6a00, #ee0979); border-radius: 10px; padding: 10px;'>
# """, unsafe_allow_html=True)
#     st.subheader("🎯 Infra Flow")
#     nodes = [
#             Node(id="DBRepo", label="DBRepo 📚", color="#f94144"),
#             Node(id="Invenio", label="Invenio 💃", color="#f3722c"),
#             Node(id="JupyterHub", label="Jupyter 💻", color="#f8961e"),
#             Node(id="GitHub", label="GitHub 🧠", color="#f9844a"),
#             Node(id="VRE", label="VRE 🧪", color="#43aa8b"),
#             Node(id="Metadata", label="Metadata 🧰", color="#577590"),
#             Node(id="Provenance JSON", label="JSON 📜", color="#277da1"),
#             Node(id="Visualization", label="Viz 🌐", color="#9b5de5")
#         ]

#     edges = [
#             Edge(source="DBRepo", target="VRE"),
#             Edge(source="Invenio", target="VRE"),
#             Edge(source="JupyterHub", target="VRE"),
#             Edge(source="GitHub", target="VRE"),
#             Edge(source="Metadata", target="Provenance JSON"),
#             Edge(source="Provenance JSON", target="Visualization"),
#             Edge(source="VRE", target="Visualization")
#         ]

#     config = Config(width=150, height=130, directed=True, physics=True)
#     agraph(nodes=nodes, edges=edges, config=config)
#     st.markdown("""
#     <script>
#     const el = document.getElementById('infraBox');
#     let offsetX = 0, offsetY = 0, isDown = false;
#     el.addEventListener('mousedown', function(e) {
#         isDown = true;
#         offsetX = e.clientX - el.getBoundingClientRect().left;
#         offsetY = e.clientY - el.getBoundingClientRect().top;
#     });
#     document.addEventListener('mouseup', () => isDown = false);
#     document.addEventListener('mousemove', function(e) {
#         if (!isDown) return;
#         el.style.left = (e.clientX - offsetX) + 'px';
#         el.style.top = (e.clientY - offsetY) + 'px';
#     });
#     </script>
#     </div>
#     """, unsafe_allow_html=True)


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
    return df[cols].to_dict(orient='records')


@st.cache_data
def load_data():
    """
    Load and flatten JSON metadata files from the MODEL_PROVENANCE directory.
    """
    files = glob.glob("MODEL_PROVENANCE/*_run_summary.json")
    rows = []
    for f in files:
        with open(f) as fh:
            summary = json.load(f)
        row = {"run_id": summary.get("run_id")}
        row.update({f"param_{k}": v for k, v in summary.get("params", {}).items()})
        row.update({f"metric_{k}": v for k, v in summary.get("metrics", {}).items()})
        row.update({f"tag_{k}": v for k, v in summary.get("tags", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)

def _get_all_features(df):
    """
    Retrieve the list of feature names from the DataFrame.
    Assumes every row has the same 'param_feature_names'.
    """
    raw = df.loc[0, 'param_feature_names']
    return ast.literal_eval(raw)

def evaluate_subset(features, test_size=0.2, random_state=42, n_estimators=200):
    """
    Train and evaluate a RandomForestClassifier on a subset of features.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    canon = _get_all_features(df)
    mapping = dict(zip(iris.feature_names, canon))
    X = X.rename(columns=mapping)
    X_sub = X[features]
    y = iris.target
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

# -------- Load the metadata (flattened like before) ----------
@st.cache_data
def load_data():
    with open("updated_metadata.json", "r") as fh:
        summary = json.load(fh)

    row = {"run_id": summary.get("run_id")}
    row.update({f"param_{k}": v for k, v in summary.get("params", {}).items()})
    row.update({f"metric_{k}": v for k, v in summary.get("metrics", {}).items()})
    row.update({f"tag_{k}": v for k, v in summary.get("tags", {}).items()})
    return pd.DataFrame([row])



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




# -------- App Layout --------

# # Sidebar navigation
# st.sidebar.title("📂 Navigation")
# page = st.sidebar.radio("Go to", [
#     "🏠 Dashboard",
#     "📁 Dataset Metadata",
#     "🧠 ML Model Metadata",
#     "📊 Model Plots",
#     "🛰️ Provenance Trace",
#     "⚠️ Deprecated Code Check"
# ])
# Sidebar navigation menu
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
            "🧭 Model-Dataset Mapping"
        ],
        icons=[
            "house", "database", "gear", "bar-chart", "globe", "link", "exclamation-triangle"
        ],
        menu_icon="cast",
        default_index=0,
    )
# Header
st.markdown("<h1 style='text-align: center;'>ML Provenance Dashboard</h1>", unsafe_allow_html=True)

# Main content switching


# Main content area
if selected == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.write("Welcome to the Dashboard!")

    st.markdown("## 👋 Welcome to the ML Provenance Dashboard")
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
    st.subheader("🔁 Infrastructure Provenance Flow")
        
        # Define nodes in the infrastructure
    nodes = [
            "DBRepo (Structured Repository)",    # 0
            "Invenio (Unstructured Repository)", # 1
            "JupyterHub (Computational Layer)",  # 2
            "GitHub (Version Control)",          # 3
            "Virtual Research Environment (VRE)",# 4
            "Metadata Extraction",               # 5
            "Provenance JSON",                   # 6
            "Interactive Visualization"          # 7
        ]
        
        # Define links (source-target-flow)
    sources = [0, 1, 2, 3, 5, 6, 4]
    targets = [4, 4, 4, 4, 6, 7, 7]
    values =  [1, 1, 1, 1, 1, 1, 1]
        
        # Build Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
    fig.update_layout(title_text="Infrastructure Provenance Flow", font_size=12)
    st.plotly_chart(fig, use_container_width=True)
###################################################################################
###################################################################################
    # st.subheader("🔁 Infrastructure Provenance Flow (Graph View)")
    
    # # Create graph
    # G = nx.DiGraph()
    
    # nodes = [
    #     "DBRepo (Structured Repository)",
    #     "Invenio (Unstructured Repository)",
    #     "JupyterHub (Computational Layer)",
    #     "GitHub (Version Control)",
    #     "Virtual Research Environment (VRE)",
    #     "Metadata Extraction",
    #     "Provenance JSON",
    #     "Interactive Visualization"
    # ]
    
    # edges = [
    #     ("DBRepo (Structured Repository)", "Virtual Research Environment (VRE)"),
    #     ("Invenio (Unstructured Repository)", "Virtual Research Environment (VRE)"),
    #     ("JupyterHub (Computational Layer)", "Virtual Research Environment (VRE)"),
    #     ("GitHub (Version Control)", "Virtual Research Environment (VRE)"),
    #     ("Metadata Extraction", "Provenance JSON"),
    #     ("Provenance JSON", "Interactive Visualization"),
    #     ("Virtual Research Environment (VRE)", "Interactive Visualization")
    # ]
    
    # # Add to graph
    # G.add_nodes_from(nodes)
    # G.add_edges_from(edges)
    
    # # Build and display interactive network
    # net = Network(height="500px", width="100%", directed=True)
    # net.from_nx(G)
    # net.set_options('''{
    #   "nodes": {
    #     "font": {"size": 14},
    #     "shape": "box",
    #     "color": {"background": "#add8e6"}
    #   },
    #   "edges": {
    #     "arrows": {"to": {"enabled": true}},
    #     "smooth": {"type": "cubicBezier"}
    #   },
    #   "physics": {
    #     "enabled": true,
    #     "barnesHut": {"gravitationalConstant": -20000}
    #   }
    # }''')
    
    # net.save_graph("infra_flow.html")
    # components.html(open("infra_flow.html", "r").read(), height=600)
    ##############################################################################################
    ##############################################################################################


    # st.subheader("🎬 Infrastructure Provenance Journey")

    # st.markdown("### Follow the data on its adventure 🚀")
    
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     st.markdown("### 📚 DBRepo")
    #     st.image("https://img.icons8.com/external-flat-juicy-fish/64/database.png", width=50)
    #     st.write("Structured storage")
    
    # with col2:
    #     st.markdown("### 🗃️ Invenio")
    #     st.image("https://img.icons8.com/external-flaticons-flat-flat-icons/64/open-archive.png", width=50)
    #     st.write("Unstructured repository")
    
    # with col3:
    #     st.markdown("### 💻 JupyterHub")
    #     st.image("https://img.icons8.com/ios/50/jupyter.png", width=50)
    #     st.write("Interactive computation")
    
    # with col4:
    #     st.markdown("### 🧠 GitHub")
    #     st.image("https://img.icons8.com/ios-filled/50/github.png", width=50)
    #     st.write("Version control")
    
    # st.markdown("↓ ↓ ↓")
    
    # col5, col6 = st.columns([1, 2])
    
    # with col5:
    #     st.markdown("### 🧰 Metadata Extraction")
    #     st.image("https://img.icons8.com/external-icongeek26-flat-icongeek26/64/extract.png", width=50)
    
    # with col6:
    #     st.markdown("### 📜 Provenance JSON → 🌐 Visualization")
    #     st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/data-visualization.png", width=60)
    #     st.write("All roads lead to insight ✨")
#############################################################################
#############################################################################


    # st.subheader("🎯 Animated Infra Flow")
    
    # nodes = [
    #     Node(id="DBRepo", label="DBRepo 📚"),
    #     Node(id="Invenio", label="Invenio 🗃️"),
    #     Node(id="JupyterHub", label="Jupyter 💻"),
    #     Node(id="GitHub", label="GitHub 🧠"),
    #     Node(id="VRE", label="VRE 🧪"),
    #     Node(id="Metadata", label="Metadata 🧰"),
    #     Node(id="Provenance JSON", label="JSON 📜"),
    #     Node(id="Visualization", label="Viz 🌐")
    # ]
    
    # edges = [
    #     Edge(source="DBRepo", target="VRE"),
    #     Edge(source="Invenio", target="VRE"),
    #     Edge(source="JupyterHub", target="VRE"),
    #     Edge(source="GitHub", target="VRE"),
    #     Edge(source="Metadata", target="Provenance JSON"),
    #     Edge(source="Provenance JSON", target="Visualization"),
    #     Edge(source="VRE", target="Visualization")
    # ]
    
    # config = Config(width=800, height=500, directed=True, physics=True)
    # agraph(nodes=nodes, edges=edges, config=config)

elif selected == "📁 Dataset Metadata":
    st.title("📁 Dataset Metadata")
    st.write("Here is the dataset metadata.")
    st.subheader("📁 Dataset Metadata")

    # Filter relevant columns
    dataset_cols = [
        "param_dataset.title", "param_dataset.doi", "param_dataset.authors",
        "param_dataset.publisher", "param_dataset.published",
        "tag_dataset_id", "tag_dataset_name", "tag_dataset_version",
        "tag_data_source", "param_database.name", "param_database.owner",
        "tag_dbrepo.repository_name"
    ]

    dataset_info = df[dataset_cols].copy()

    # Dropdown to select dataset (optional, if more than one)
    dataset_names = dataset_info["tag_dataset_name"].dropna().unique()

    selected_dataset = st.selectbox("Choose dataset", dataset_names)

    filtered_df = dataset_info[dataset_info["tag_dataset_name"] == selected_dataset]

    if not filtered_df.empty:
        st.write("### Selected Dataset Metadata")
        st.dataframe(filtered_df.T)
    else:
        st.warning("No matching dataset found.")


elif selected == "🧠 ML Model Metadata":
    st.title("🧠 ML Model Metadata")
    st.write("Details about the ML model.")

    st.subheader("🧠 ML Training Configuration")

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

            st.markdown("#### 🛢️ DBRepo Lineage (Optional)")
            st.dataframe(run_df.filter(
                regex=r'^metric_dbrepo\..*'
            ).T)



elif selected == "📊 Model Plots":
    st.title("📊 Model Plots")
    st.write("Visualizations of the model.")
    st.subheader("📊 Model Explainability & Evaluation Plots")

    plot_dir = "plots"  # Adjust if you store plots elsewhere

    plot_options = {
        "Feature Importances": "RandomForest_Iris_v20250424_111946/feature_importances.png",
        "Confusion Matrix": "RandomForest_Iris_v20250424_111946/confusion_matrix.png",
        "SHAP Summary": "RandomForest_Iris_v20250424_111946/shap_summary.png",
        "ROC Curve (Class 0)": "RandomForest_Iris_v20250424_111946/roc_curve_cls_0.png",
        "Precision-Recall (Class 0)": "RandomForest_Iris_v20250424_111946/pr_curve_cls_0.png"
    }

    selected_plot = st.selectbox("Choose a plot to view", list(plot_options.keys()))

    try:
        plot_path = os.path.join(plot_dir, plot_options[selected_plot])
        # st.image(plot_path, caption=selected_plot, width=600)
        plot_width = st.slider("Adjust plot width", 400, 1000, 600)
        st.image(plot_path, caption=selected_plot, width=plot_width)

        # Optional: add explanation under plot
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
    except FileNotFoundError:
        st.warning("Plot not found. Please ensure the image exists in the correct directory.")


elif selected == "🛰️ Provenance Trace":
    st.title("🛰️ Provenance Trace")
    st.write("Tracing the provenance.")

    st.subheader("🛰️ Provenance Trace")

    # Select a use case
    use_case_name = st.selectbox("Select a Use Case", list(USE_CASES.keys()))
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
    st.write("Mapping between models and datasets.")
    st.subheader("🔗 Model to Dataset Mapping")

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
    st.write("Checking for deprecated code.")
    st.subheader("⚠️ Deprecated Code Check")
    
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

# if page == "🏠 DASHBOARD":

#         st.markdown("## 👋 Welcome to the ML Provenance Dashboard")
#         st.markdown("""
#         This tool is designed for researchers to interactively inspect:
#         - 🧬 Dataset metadata  
#         - 🧠 Training configuration  
#         - 🛰️ Provenance trace  
#         - ⚠️ Deprecated code usage  
        
#         Use the left menu to navigate between sections.
#         """)
        
#         st.markdown("---")
#         st.subheader("🔁 Infrastructure Provenance Flow")
        
#         # Define nodes in the infrastructure
#         nodes = [
#             "DBRepo (Structured Repository)",    # 0
#             "Invenio (Unstructured Repository)", # 1
#             "JupyterHub (Computational Layer)",  # 2
#             "GitHub (Version Control)",          # 3
#             "Virtual Research Environment (VRE)",# 4
#             "Metadata Extraction",               # 5
#             "Provenance JSON",                   # 6
#             "Interactive Visualization"          # 7
#         ]
        
#         # Define links (source-target-flow)
#         sources = [0, 1, 2, 3, 5, 6, 4]
#         targets = [4, 4, 4, 4, 6, 7, 7]
#         values =  [1, 1, 1, 1, 1, 1, 1]
        
#         # Build Sankey Diagram
#         fig = go.Figure(data=[go.Sankey(
#             node=dict(
#                 pad=15,
#                 thickness=20,
#                 line=dict(color="black", width=0.5),
#                 label=nodes
#             ),
#             link=dict(
#                 source=sources,
#                 target=targets,
#                 value=values
#             )
#         )])
        
#         fig.update_layout(title_text="Infrastructure Provenance Flow", font_size=12)
#         st.plotly_chart(fig, use_container_width=True)


# elif page == "📁 Dataset Metadata":
#     st.subheader("📁 Dataset Metadata")

#     # Filter relevant columns
#     dataset_cols = [
#         "param_dataset.title", "param_dataset.doi", "param_dataset.authors",
#         "param_dataset.publisher", "param_dataset.published",
#         "tag_dataset_id", "tag_dataset_name", "tag_dataset_version",
#         "tag_data_source", "param_database.name", "param_database.owner",
#         "tag_dbrepo.repository_name"
#     ]

#     dataset_info = df[dataset_cols].copy()

#     # Dropdown to select dataset (optional, if more than one)
#     dataset_names = dataset_info["tag_dataset_name"].dropna().unique()

#     selected_dataset = st.selectbox("Choose dataset", dataset_names)

#     filtered_df = dataset_info[dataset_info["tag_dataset_name"] == selected_dataset]

#     if not filtered_df.empty:
#         st.write("### Selected Dataset Metadata")
#         st.dataframe(filtered_df.T)
#     else:
#         st.warning("No matching dataset found.")

# elif page == "🧠 ML Model Metadata":
    # st.subheader("🧠 ML Training Configuration")

    # run_ids = df['run_id'].dropna().unique()
    # selected_run = st.selectbox("Select a Run ID", run_ids)
    # run_df = df[df["run_id"] == selected_run]

    # if run_df.empty:
    #     st.warning("No matching run found.")
    # else:
    #     tab1, tab2, tab3, tab4 = st.tabs([
    #         "🛠️ Hyperparameters", "💻 Environment", "📊 Dataset Sampling", "📈 Metrics"
    #     ])

    #     with tab1:
    #         st.write("### Training Hyperparameters")
    #         st.dataframe(run_df.filter(
    #             regex=r'^param_(criterion|max_depth|max_features|min_samples_split|min_samples_leaf|n_estimators|bootstrap|warm_start|oob_score|random_state)'
    #         ).T)

    #     with tab2:
    #         st.write("### Environment Info")
    #         st.dataframe(run_df.filter(
    #             regex=r'^param_(numpy_version|pandas_version|python_version|sklearn_version|matplotlib_version|seaborn_version|os_platform)'
    #         ).T)

    #     with tab3:
    #         st.write("### Dataset Size & Sampling")
    #         st.dataframe(run_df.filter(
    #             regex=r'^param_(n_records|n_features|n_train_samples|n_test_samples|test_size|max_samples)'
    #         ).T)

    #     with tab4:
    #         st.write("### Model Performance Metrics")

    #         col1, col2 = st.columns(2)

    #         with col1:
    #             st.markdown("#### ✅ Evaluation (Test Set)")
    #             st.dataframe(run_df.filter(
    #                 regex=r'^metric_(accuracy$|f1_score_X_test|precision_score_X_test|recall_score_X_test|roc_auc_score_X_test)'
    #             ).T)

    #         with col2:
    #             st.markdown("#### 🏋️ Training Set")
    #             st.dataframe(run_df.filter(
    #                 regex=r'^metric_training_'
    #             ).T)

    #         st.markdown("#### 🛢️ DBRepo Lineage (Optional)")
    #         st.dataframe(run_df.filter(
    #             regex=r'^metric_dbrepo\..*'
    #         ).T)


# elif page == "🛰️ Provenance Trace":
#     st.subheader("🛰️ Provenance Trace")

#     # Select a use case
#     use_case_name = st.selectbox("Select a Use Case", list(USE_CASES.keys()))
#     use_case = USE_CASES[use_case_name]

#     # Collect required parameters
#     params = {}
#     for param in use_case['required_params']:
#         if param == 'feature':
#             all_features = _get_all_features(df)
#             selected_feature = st.selectbox("Select a Feature", all_features)
#             params['feature'] = selected_feature
#         elif param == 'features':
#             all_features = _get_all_features(df)
#             selected_features = st.multiselect("Select Features", all_features)
#             params['features'] = selected_features
#         elif param == 'threshold':
#             threshold = st.slider("Set Accuracy Threshold", min_value=0.0, max_value=1.0, value=0.95)
#             params['threshold'] = threshold
#         else:
#             param_value = st.text_input(f"Enter value for {param}")
#             params[param] = param_value

#     # Collect optional parameters
#     for param in use_case['optional_params']:
#         if param == 'run_id':
#             run_ids = df['run_id'].dropna().unique()
#             selected_run_id = st.selectbox("Select a Run ID (Optional)", run_ids)
#             params['run_id'] = selected_run_id
#         else:
#             param_value = st.text_input(f"Enter value for {param} (Optional)")
#             params[param] = param_value

#     # Execute the selected use case
#     if st.button("Run Use Case"):
#         try:
#             result = use_case['func'](df, **params)
#             if isinstance(result, pd.DataFrame):
#                 st.dataframe(result)
#             elif isinstance(result, list):
#                 st.json(result)
#             elif isinstance(result, dict):
#                 st.json(result)
#             else:
#                 st.write(result)
#         except Exception as e:
#             st.error(f"An error occurred: {e}")


# elif page == "⚠️ Deprecated Code Check":
    # st.subheader("⚠️ Deprecated Code Check")
    
    #     # Input for deprecated commits
    # deprecated_commits_input = st.text_area(
    #         "Enter deprecated commit hashes (one per line):",
    #         height=100
    #     )
    
    # if deprecated_commits_input:
    #         deprecated_commits = [line.strip() for line in deprecated_commits_input.strip().split('\n') if line.strip()]
    #         if st.button("Run Check"):
    #             try:
    #                 results = detect_deprecated_code(df, deprecated_commits=deprecated_commits)
    #                 if results:
    #                     st.write("### Runs using deprecated commits:")
    #                     st.dataframe(pd.DataFrame(results))
    #                 else:
    #                     st.success("No runs found with deprecated commits.")
    #             except Exception as e:
    #                 st.error(f"An error occurred: {e}")
    # else:
    #         st.info("Please enter at least one deprecated commit hash.")

# elif page == "📊 Model Plots":
    # st.subheader("📊 Model Explainability & Evaluation Plots")

    # plot_dir = "plots"  # Adjust if you store plots elsewhere

    # plot_options = {
    #     "Feature Importances": "RandomForest_Iris_v20250424_111946/feature_importances.png",
    #     "Confusion Matrix": "RandomForest_Iris_v20250424_111946/confusion_matrix.png",
    #     "SHAP Summary": "RandomForest_Iris_v20250424_111946/shap_summary.png",
    #     "ROC Curve (Class 0)": "RandomForest_Iris_v20250424_111946/roc_curve_cls_0.png",
    #     "Precision-Recall (Class 0)": "RandomForest_Iris_v20250424_111946/pr_curve_cls_0.png"
    # }

    # selected_plot = st.selectbox("Choose a plot to view", list(plot_options.keys()))

    # try:
    #     plot_path = os.path.join(plot_dir, plot_options[selected_plot])
    #     # st.image(plot_path, caption=selected_plot, width=600)
    #     plot_width = st.slider("Adjust plot width", 400, 1000, 600)
    #     st.image(plot_path, caption=selected_plot, width=plot_width)

    #     # Optional: add explanation under plot
    #     if "Feature Importances" in selected_plot:
    #         st.markdown("**Interpretation:** Shows which features contribute most to predictions.")
    #     elif "SHAP" in selected_plot:
    #         st.markdown("**Interpretation:** SHAP summary plots show feature impact and distribution.")
    #     elif "ROC" in selected_plot:
    #         st.markdown("**Interpretation:** ROC curves visualize classifier trade-off between sensitivity and specificity.")
    #     elif "Precision-Recall" in selected_plot:
    #         st.markdown("**Interpretation:** Precision-Recall curves help understand classifier performance on imbalanced data.")
    #     elif "Confusion" in selected_plot:
    #         st.markdown("**Interpretation:** The confusion matrix shows how many predictions were correct or misclassified.")
    # except FileNotFoundError:
    #     st.warning("Plot not found. Please ensure the image exists in the correct directory.")

# elif page == "Model-Dataset Mapping":
#     st.subheader("🔗 Model to Dataset Mapping")

#     try:
#         results = map_model_dataset(df)
#         if results:
#             st.dataframe(pd.DataFrame(results))
#         else:
#             st.warning("No model-dataset mappings found.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")