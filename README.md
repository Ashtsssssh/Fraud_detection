# AML Transaction Monitoring System

A machine learning system for detecting anti-money laundering (AML) transactions, built from scratch to study feature engineering, class imbalance, and model interpretability.

## Architecture
- **Data Processing**: Pandas
- **Machine Learning**: LightGBM (with threshold optimization)
- **Interpretability**: SHAP
- **Dashboard**: Streamlit

## Getting Started

1. Install requirements:
`pip install -r req.txt`

2. Download the Kaggle dataset (`SAML-D.csv`) and place it in `data/raw/`

3. Run the pipeline to train the model:
`python run_pipeline.py`

4. Launch the dashboard:
`streamlit run dashboard/app.py`
