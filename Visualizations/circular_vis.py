import dash
from dash import dcc, html, Input, Output
import dash_cytoscape as cyto
import json

# Load JSON data
json_data = {
  "db_repo_structured_db": {
    "identifier": "Globally Unique ID for dataset from the registry (FAIR-DATA, DataCite)Example: DOI (Digital Object Identifier), ORCID (for researchers), UUIDs (Universally Unique Identifiers), and Handles (e.g., Handle.net). for datatset, Zonodo, Datacite f1",
    "title": "Dataset title (FAIR-DATA, DataCite)",
    "description": "Brief description of dataset (FAIR-DATA, Dublin Core)",
    "keywords": ["Relevant search terms (FAIR-DATA)"],
    "creators": "Person or organization that created dataset (PROV-O, DataCite)",
    "created_date": "Date dataset was created (FAIR-DATA, DataCite)",
    "modified_date": "Date of last modification (FAIR-DATA, DataCite)",
    "provenance": "History of dataset, modifications, and ownership (PROV-O)",
    "access_rights": "Who can access the dataset (FAIR-DATA, Dublin Core)a1",
    "license": "Usage license (FAIR-DATA, DataCite)",
    "source": "Original source if derived from another dataset (PROV-O)",
    "relations": ["Links to related datasets (FAIR-DATA, DataCite)"],
    "storage_location": "Physical or digital location of dataset (FAIR-DATA)",
    "metadata_schema": "Schema used for metadata (FAIR-DATA, DataCite)",
    "format": "File format (FAIR-DATA, Dublin Core)",
    "size": "Size of dataset (FAIR-DATA)",
    "checksum": "Integrity check value (FAIR-DATA)",
    "modification_history": "Track dataset changes (PROV-O)",
    "funding": "Funding source (FAIR-DATA)",
    "contributors": ["List of contributors (FAIR-DATA, PROV-O)"],
    "derived_from": "Parent dataset if applicable (PROV-O)",
    "version": "Dataset version (FAIR-DATA, DataCite)",
    "contact_point": "Responsible contact (FAIR-DATA)",
    "FAIR_compliance_score": "Self-evaluation of FAIR principles (FAIR-DATA)",
    "documentation": "Link to dataset documentation (FAIR-DATA)f2",
    "reference_strategy":"FAIR, Datacite, PROVO, DUblincore",
    "dataset": {
        "collection_method": "How the data was collected (e.g., surveys, sensors, API, manual entry)",
        "data_structure": "Description of dataset structure (e.g., tabular, relational database, hierarchical, graph-based)",
        "num_records": "Total number of records (rows) in the dataset",
        "num_columns": "Total number of features (columns) in the dataset",
        "column_types": {
            "categorical": ["List of categorical columns"],
            "numerical": ["List of numerical columns"],
            "text": ["List of text-based columns"],
            "date_time": ["List of datetime columns"],
            "boolean": ["List of boolean columns"]
        },
        "summary_statistics": {
            "numerical": {
                "mean": "Mean of numerical columns",
                "median": "Median of numerical columns",
                "std_dev": "Standard deviation of numerical columns",
                "min": "Minimum values per numerical column",
                "max": "Maximum values per numerical column"
            },
            "categorical": {
                "unique_values": "Unique values per categorical column",
                "most_frequent": "Most common category per categorical column"
            }
        },
        "missing_data": {
            "total_missing_values": "Total number of missing values in the dataset",
            "missing_columns": ["List of columns with missing values"],
            "missing_handling": "How missing data was handled (e.g., imputation, removal, left as-is)"
        },
        "outliers": {
            "columns_with_outliers": ["List of columns with detected outliers"],
            "handling_method": "How outliers were treated (e.g., capped, removed, normalized)"
        },
        "sampling_strategy": "How data was sampled (e.g., random sampling, stratified sampling)",
        "data_quality": "Assessment of dataset quality, completeness, and validation checks performed",
        "bias_check": "Any known biases in the dataset (e.g., demographic imbalance, selection bias)",
        "anonymization": "If data is anonymized, the techniques used (e.g., k-anonymity, differential privacy)",
        "geospatial_info": "Geographical data details (e.g., GPS coordinates, country codes)",
        "temporal_coverage": "Time period covered by the dataset (e.g., 2015-2023)",
        "usage_guidelines": "Best practices for using this dataset, recommended tools for analysis",
        "known_issues": "Any known limitations, biases, or missing values in the dataset",
        "citations": "References to papers, projects, or research that have used this dataset",
        "ethics_approval": "If the dataset involves human data, details on IRB/ethics board approval"
    }
  },
  "unstructured_repository_short": {
    "identifier": "Unique ID for dataset (FAIR-DATA)",
    "title": "Dataset title (FAIR-DATA, Dublin Core)",
    "description": "Brief description (FAIR-DATA, Dublin Core)",
    "keywords": ["Relevant search terms (FAIR-DATA)"],
    "creator": "Person/organization that created dataset (PROV-O, DataCite)",
    "created_date": "Creation date (FAIR-DATA)",
    "modified_date": "Last modification date (FAIR-DATA)",
    "provenance": "History of dataset, modifications, ownership (PROV-O)",
    "access_rights": "Access permissions (FAIR-DATA, Dublin Core)",
    "license": "Usage license (FAIR-DATA, DataCite)",
    "source": "Original source (PROV-O)",
    "relations": ["Links to related datasets (FAIR-DATA, DataCite)"],
    "storage_location": "Physical/digital location (FAIR-DATA)",
    "format": "File format (FAIR-DATA, Dublin Core)",
    "size": "Size of dataset (FAIR-DATA)",
    "modification_history": "Track changes (PROV-O)",
    "derived_from": "Parent dataset if applicable (PROV-O)",
    "FAIR_compliance_score": "Self-evaluation of FAIR principles (FAIR-DATA)",
    "documentation": "Link to dataset documentation (FAIR-DATA)"
  },
  "unstructured_repository": {
    "identifier": "Unique ID for stored artifact (UUID, DOI, Handle) (FAIR-DATA, DataCite, PROV-O)",
    "title": "Title of stored artifact (e.g., model name, log file title) (FAIR-DATA, Dublin Core)",
    "description": "Brief description of the stored artifact (FAIR-DATA, Dublin Core)",
    "keywords": ["Relevant search terms (FAIR-DATA)"],
    "creator": {
        "name": "Person/organization that created the artifact",
        "role": "Role (e.g., Data Scientist, ML Engineer, System Log Manager) (PROV-O)",
        "affiliation": "Institution or company (FAIR-DATA)"
    },
    "created_date": "Date when artifact was created (FAIR-DATA, PROV-O)",
    "modified_date": "Last modification date (FAIR-DATA, PROV-O)",
    "artifact_type": "Type of stored file (e.g., ML model, log file, dataset snapshot, documentation) (FAIR-DATA)",
    "licesnses": "Type of licesnses",
    "publish": {
    "published_by":"person or org who published",
    "published_date":"person or org who published",
    },
    "provenance": {
        "generated_by": "Process that created this artifact (e.g., ML training pipeline, system logging) (PROV-O)",
        "version": "Artifact version identifier (e.g., v1.0, v2.3) (FAIR-DATA, DataCite)",
        "lineage": "List of related artifacts that influenced this (e.g., previous model version, training dataset) (PROV-O)"
    },
    "access_rights": "Who can access this artifact (Public, Restricted, Private) (FAIR-DATA, Dublin Core)",
    "license": "Usage license (e.g., MIT, Apache 2.0, Creative Commons) (FAIR-DATA, DataCite)",
    "source": "If derived, original source or upstream reference (PROV-O, DataCite)",
    "relations": ["Links to related artifacts (e.g., models trained on same dataset, log files for this model) (FAIR-DATA, DataCite)"],
    "storage_location": {
        "primary": "Path where the artifact is stored (e.g., S3 bucket, Invenio repository, institutional storage) (FAIR-DATA)",
        "backup": "Backup location if applicable",
        "checksum": "SHA-256 or MD5 checksum to ensure data integrity (FAIR-DATA)"
    },
    "format": "File format (e.g., .h5, .onnx, .json, .log, .txt) (FAIR-DATA, Dublin Core)",
    "size": "File size in MB, GB (FAIR-DATA)",
    "modification_history": [
        {
            "modified_date": "Timestamp of modification",
            "modified_by": "Person/system modifying it",
            "change_summary": "Description of what was modified"
        }
    ],
    "derived_from": "Reference to parent artifact (e.g., previous model version, dataset) (PROV-O)",
    "FAIR_compliance_score": "Self-assessed compliance with FAIR principles (FAIR-DATA)",
    "documentation": "Link to documentation (e.g., README, API reference) (FAIR-DATA)",
    "retention_policy": {
        "retention_period": "How long this artifact will be stored (e.g., 5 years, indefinitely)",
        "archival_status": "Whether the file is archived or active"
    },
    "audit_log": [
        {
            "action": "Created/Modified/Reviewed",
            "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
            "performed_by": "User/System that made the change"
        }
    ]
},

  "machine_learning_metadata": {
    "experiment_id": "Unique ID for experiment (FAIR-ML, MLflow, PROV-O)",
    "model_name": "Name of the trained model (FAIR-ML, MLflow)",
    "algorithm": "Algorithm used (e.g., Random Forest, CNN, LSTM) (FAIR-ML)",
    "dataset_used": "Reference to dataset ID or DOI (FAIR-ML, PROV-O, DataCite)",
    "training_date": "Date when the model was trained in YYYY-MM-DD format (FAIR-ML)",
    "trained_by": "Person/organization responsible for training (PROV-O)",
    "data_preprocessing": {
        "missing_values": {
            "strategy": "How missing data was handled (e.g., imputation, removal, mean/median fill, interpolation)",
            "affected_columns": ["List of columns with missing values"],
            "imputation_method": "If imputation was used, specify technique (e.g., KNN, mean, regression)"
        },
        "feature_scaling": {
            "scaling_method": "Scaling method applied (e.g., MinMax, Standardization, Robust Scaling)",
            "applied_columns": ["List of columns where scaling was applied"]
        },
        "encoding": {
            "categorical_encoding": "Method used (e.g., One-Hot Encoding, Label Encoding, Embedding Layers)",
            "affected_columns": ["List of categorical columns transformed"]
        },
        "outlier_handling": {
            "strategy": "How outliers were handled (e.g., winsorization, removal, log transformation)",
            "affected_columns": ["List of columns where outliers were detected"],
            "detection_method": "Method used for outlier detection (e.g., Z-score, IQR, DBSCAN)"
        },
        "data_transformation": {
            "applied_methods": ["Log transformation", "Polynomial features", "PCA"],
            "reason_for_transformation": "Purpose (e.g., reducing dimensionality, normalizing skewed data)"
        },
        "data_splits": {
            "train_test_split": "Percentage split between training and test sets (e.g., 80-20, 70-30)",
            "cross_validation_folds": "Number of folds used in cross-validation (e.g., k=5)"
        }
    },
    "hyperparameters": {
        "learning_rate": 0.0,
        "batch_size": 0,
        "epochs": 0,
        "optimizer": "Adam",
        "loss_function": "Cross-entropy"
    },
    "performance_metrics": {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc_roc": 0.0,
        "loss":  0.0
    },
    "model_explainability": {
        "explainability_report": "SHAP, LIME, Attention Weights report (FAIR-ML)",
        "feature_importance": ["List of most important features impacting model predictions"],
        "explainability_score": "Quantitative score measuring model transparency (FAIR-ML)"
    },
    "dependencies": ["List of required libraries, e.g., TensorFlow, Scikit-Learn, PyTorch (FAIR-ML, MLflow)"],
    "environment": "Details of software/hardware used, including GPU, CPU, RAM, OS (FAIR-ML, MLflow)",
    "model_artifact_location": "Storage path of trained model (FAIR-ML, MLflow)",
    "license": "License under which the model can be used (e.g., MIT, Apache 2.0) (FAIR-ML, DataCite)",
    "source_code": "GitHub/Bitbucket link to model training and inference scripts (FAIR-ML, PROV-O)",
    "training_logs": "Logs generated during training, including time per epoch and loss trends (FAIR-ML, MLflow)",
    "previous_model": "If model is an updated version, reference to prior model (PROV-O)",
    "next_model": "Future versions of the model, if applicable (PROV-O)",
    "derived_from": "Reference to original dataset or prior models (PROV-O, FAIR-ML)",
    "model_card": "Standardized documentation of model details (FAIR-ML, MLflow)",
    "validation_results": {
        "dataset_id": "ID/DOI of validation dataset",
        "validation_metrics": {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
    },
    "deployment_status": "Model deployment details (e.g., production, staging, inactive) (FAIR-ML)",
    "deployment_endpoint": "URL or API endpoint for deployed model",
    "reproducibility_guidelines": "Detailed steps for re-running experiment and training pipeline (FAIR-ML)",
    "bias_analysis": {
        "bias_check_method": "Techniques used for bias detection (e.g., disparate impact, equalized odds)",
        "identified_bias": "Summary of detected biases in dataset or model predictions",
        "mitigation_strategies": "Steps taken to reduce bias (e.g., reweighting, adversarial debiasing)"
    }
}
,
  "github_repository_metadata_short_version": {
    "repo_id": "Unique identifier (FAIR-DATA, GitHub API)",
    "repo_name": "Name of repository (GitHub API)",
    "owner": "Repository owner (GitHub API)",
    "created_date": "Date repo was created (FAIR-DATA, GitHub API)",
    "modified_date": "Last update date (FAIR-DATA, GitHub API)",
    "contributors": ["List of contributors (FAIR-DATA, GitHub API)"],
    "license": "Repository license (FAIR-DATA, GitHub API)",
    "visibility": "Public/private (GitHub API)",
    "source_code": "Main source code URL (FAIR-DATA, GitHub API)",
    "dependencies": ["List of dependencies (FAIR-DATA, GitHub API)"],
    "commits": "Number of commits (GitHub API)",
    "pull_requests": "Number of PRs merged (GitHub API)",
    "workflow_runs": "List of automated CI/CD runs (GitHub API)",
    "issues": "Open/closed issues (GitHub API)",
    "forks": "Number of forks (GitHub API)",
    "stars": "Number of stars (GitHub API)",
    "watchers": "Number of watchers (GitHub API)",
    "readme": "Link to README (FAIR-DATA, GitHub API)",
    "releases": "List of releases (GitHub API)",
    "security_scans": "Vulnerability scans (GitHub API)",
    "modification_history": "Track changes (PROV-O)",
    "derived_from": "If repo is forked from another (PROV-O)",
    "reproducibility_guidelines": "Steps for reproducing results (FAIR-ML)"
  },




  "github_repository_metadata_longer_version": {
    "repo_id": "Unique identifier assigned by GitHub API (FAIR-DATA, GitHub API)",
    "repo_name": "Name of the repository (GitHub API)",
    "owner": {
        "username": "Owner's GitHub username",
        "profile_url": "URL to GitHub profile"
    },
    "created_date": "Date repository was created (YYYY-MM-DD) (FAIR-DATA, GitHub API)",
    "modified_date": "Last update timestamp (YYYY-MM-DD) (FAIR-DATA, GitHub API)",
    "primary_language": "Main programming language used (GitHub API)",
    "topics": ["List of GitHub topics/tags associated with repo (GitHub API)"],
    "description": "Short description of repository (GitHub API, FAIR-DATA)",
    "contributors": [
        {
            "username": "GitHub username",
            "profile_url": "GitHub profile link",
            "contributions": "Number of contributions"
        }
    ],
    "license": {
        "license_name": "Repository license (e.g., MIT, Apache 2.0, GPL)",
        "license_url": "URL to license file"
    },
    "visibility": "Whether repository is Public/Private (GitHub API)",
    "fork_status": {
        "is_forked": "Boolean value indicating if repo is a fork",
        "original_repo": "URL of original repository if forked"
    },
    "source_code": {
        "repo_url": "Main repository URL",
        "default_branch": "Default branch name (e.g., main, master)"
    },
    "dependencies": ["List of dependencies extracted from package managers (e.g., requirements.txt, package.json, Pipfile)"],
    "commits": {
        "total_commits": "Number of commits in the repository",
        "latest_commit_hash": "SHA hash of the latest commit",
        "latest_commit_timestamp": "Date of latest commit (YYYY-MM-DD)",
        "commit_authors": ["List of authors who have committed"]
    },
    "pull_requests": {
        "total_prs": "Number of pull requests created",
        "merged_prs": "Number of PRs merged",
        "open_prs": "Number of open PRs",
        "closed_prs": "Number of closed PRs"
    },
    "workflow_runs": {
        "ci_cd_provider": "CI/CD tool used (e.g., GitHub Actions, Travis CI, Jenkins)",
        "total_runs": "Number of workflow runs",
        "last_run_status": "Success/Failure/Cancelled",
        "last_run_timestamp": "Timestamp of last CI/CD execution"
    },
    "issues": {
        "total_issues": "Total number of issues created",
        "open_issues": "Number of open issues",
        "closed_issues": "Number of closed issues"
    },
    "discussion_threads": {
        "total_discussions": "Total number of discussions in GitHub Discussions",
        "open_discussions": "Number of open discussions",
        "resolved_discussions": "Number of resolved discussions"
    },
    "forks": "Total number of forks (GitHub API)",
    "stars": "Total number of stars (GitHub API)",
    "watchers": "Total number of watchers (GitHub API)",
    "readme": {
        "readme_url": "Link to README file",
        "readme_last_updated": "Timestamp of last update to README"
    },
    "code_quality": {
        "linting_status": "Results from code linting tools (e.g., Flake8, ESLint, Pylint)",
        "test_coverage": "Percentage of test coverage from CI/CD pipeline",
        "static_analysis_tools": ["List of tools used (e.g., SonarQube, Bandit, CodeQL)"]
    },
    "security_scans": {
        "vulnerability_scan_results": "Summary of security scan (e.g., GitHub Dependabot alerts)",
        "critical_vulnerabilities": "Number of critical security vulnerabilities",
        "high_vulnerabilities": "Number of high severity security vulnerabilities",
        "medium_vulnerabilities": "Number of medium severity security vulnerabilities"
    },
    "documentation": {
        "docs_url": "Link to additional documentation or Wiki",
        "doc_coverage": "Estimated documentation completeness percentage",
        "api_documentation": "Link to API reference, if applicable"
    },
    "releases": {
        "latest_release_version": "Version number of latest release",
        "latest_release_date": "Date of latest release",
        "release_notes_url": "Link to latest release notes"
    },
    "modification_history": {
        "change_log_url": "Link to CHANGELOG file",
        "release_history": ["List of previous versions and their changes"]
    },
    "derived_from": "If repository is derived from another project, link to original repo (PROV-O)",
    "reproducibility_guidelines": {
        "setup_steps": "Instructions for setting up the repository locally",
        "dataset_references": ["Links to datasets used in the repository"],
        "hardware_requirements": "Minimum hardware needed for running the code",
        "docker_setup": "Dockerfile or docker-compose configuration, if available",
        "virtual_environment": "Virtual environment setup (e.g., Conda, venv, Poetry)"
    },
    "collaboration_tools": {
        "discussion_enabled": "Whether GitHub Discussions is enabled",
        "projects_enabled": "Whether GitHub Projects is used for issue tracking",
        "wiki_enabled": "Whether the repository has a GitHub Wiki"
    },





    "file_metadata": {
    "file_id": "Unique identifier for this metadata file (UUID, DOI)",
    "file_name": "Name of the metadata file (e.g., dataset_metadata.json)",
    "file_version": "Version of this metadata file (e.g., v1.0, v2.1)",
    "generated_by": {
        "name": "Full name of person/system generating the file",
        "organization": "Affiliated institution or company",
        "role": "Role of the person (e.g., Data Engineer, Researcher, ML Engineer)",
        "contact_email": "Email of the responsible person"
    },
    "generated_on": "Date and time file was created (ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ)",
    "last_modified": "Timestamp of last modification",
    "authorization": {
        "approved_by": "Name of person or system that authorized this metadata",
        "approval_date": "Date of approval (YYYY-MM-DD)",
        "approval_status": "Approval status (e.g., Pending, Approved, Rejected)",
        "access_control": "Who can modify this file (e.g., Admins, Specific Users, Open Access)"
    },
    "storage": {
        "storage_location": "File storage path (e.g., local path, cloud URL, database ID)",
        "storage_type": "Type of storage (e.g., AWS S3, GitHub, Institutional Server)",
        "backup_location": "Backup path or replication details",
        "encryption_status": "Whether the file is encrypted (True/False)",
        "checksum": "SHA-256 or MD5 checksum for integrity verification"
    },
    "audit_trail": [
        {
            "action": "Created/Modified/Reviewed",
            "timestamp": "Date and time of action",
            "performed_by": "Who performed the action"
        }
    ],
    "retention_policy": {
        "retention_period": "How long this file will be stored (e.g., 5 years, indefinitely)",
        "deletion_date": "Planned deletion date, if applicable",
        "archival_status": "Whether the file is archived or active"
    },
    "compliance": {
        "FAIR_compliance": "FAIR assessment score (if applicable)",
        "GDPR_status": "Whether the data follows GDPR compliance (True/False)",
        "data_ethics_review": "Ethics board approval if required"
    }
}


},
"file_metadata_short": {
    "file_id": "UUID or DOI of this metadata file",
    "file_name": "dataset_metadata.json",
    "file_version": "v1.0",
    "generated_by": {
        "name": "Creator’s Name",
        "organization": "Institution/Company",
        "role": "Researcher/Data Engineer",
        "contact_email": "email@example.com"
    },
    "generated_on": "YYYY-MM-DDTHH:MM:SSZ",
    "last_modified": "YYYY-MM-DDTHH:MM:SSZ",
    "authorization": {
        "approved_by": "Approver’s Name",
        "approval_date": "YYYY-MM-DD",
        "status": "Approved/Pending"
    },
    "storage": {
        "location": "Cloud URL / Local Path",
        "type": "AWS S3 / GitHub / Institutional Server",
        "backup": "Backup location",
        "checksum": "SHA-256 hash"
    },
    "audit_trail": [
        {"action": "Created/Modified", "timestamp": "YYYY-MM-DDTHH:MM:SSZ", "by": "User Name"}
    ],
    "retention_policy": {
        "duration": "5 years / Indefinite",
        "deletion_date": "YYYY-MM-DD"
    },
    "compliance": {
        "FAIR_score": "Assessment score",
        "GDPR_status": "Compliant/Not Compliant"
    }
}


}

# Convert JSON to Cytoscape nodes/edges
def generate_cytoscape_elements(data, parent_id="root", level=0, max_depth=3):
    elements = []
    if isinstance(data, dict) and level < max_depth:
        for key, value in data.items():
            node_id = f"{parent_id}-{key}"
            elements.append({"data": {"id": node_id, "label": key}, "classes": f"level-{level}"})
            elements.append({"data": {"source": parent_id, "target": node_id}})
            elements.extend(generate_cytoscape_elements(value, node_id, level+1, max_depth))
    elif isinstance(data, list) and level < max_depth:
        for i, item in enumerate(data):
            node_id = f"{parent_id}-{i}"
            elements.append({"data": {"id": node_id, "label": f"{parent_id}[{i}]"}, "classes": f"level-{level}"})
            elements.append({"data": {"source": parent_id, "target": node_id}})
            elements.extend(generate_cytoscape_elements(item, node_id, level+1, max_depth))
    else:
        leaf_id = f"{parent_id}-value"
        elements.append({"data": {"id": leaf_id, "label": str(data)}, "classes": f"level-{level}"})
        elements.append({"data": {"source": parent_id, "target": leaf_id}})
    return elements

elements = [{"data": {"id": "root", "label": "JSON Data"}}] + generate_cytoscape_elements(json_data)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive JSON Viewer"),
    
    # JSON Tree Visualization with improved layout
    cyto.Cytoscape(
        id='json-tree',
        layout={'name': 'circle', 'radius': 100, 'animate': True},  # More organized and less crowded layout
        style={'width': '100%', 'height': '500px'},
        elements=elements,
        stylesheet=[
            {"selector": "node", "style": {"label": "data(label)", "background-color": "#0074D9"}},
            {"selector": "edge", "style": {"line-color": "#ccc"}},
            {"selector": ".level-0", "style": {"background-color": "#FF4136"}},
            {"selector": ".level-1", "style": {"background-color": "#2ECC40"}},
            {"selector": ".level-2", "style": {"background-color": "#FFDC00"}}
        ]
    ),

    # Detail Panel
    html.Div(id="node-detail", style={"padding": "20px", "border": "1px solid #ddd", "marginTop": "20px"})
])

@app.callback(
    Output("node-detail", "children"),
    Input("json-tree", "tapNodeData")
)
def display_node_data(data):
    if not data:
        return "Click on a node to see details"
    return html.Pre(json.dumps(data, indent=2))

if __name__ == '__main__':
    app.run_server(debug=True)