"""Android Games Hit Prediction — Streamlit dashboard.
Run: streamlit run app.py
Models must be exported from the trained notebook using export_models.py instructions.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Android Games | ML Studio', page_icon='🎮', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.5rem;max-width: 1300px}
[data-testid="stMetric"] {border:1px solid #2b3b53;border-radius:13px;padding:18px;background:#101b2c}
.hero {padding:28px 32px;border-radius:19px;background:linear-gradient(110deg,#112c4a,#13253c 55%,#25365b);border:1px solid #355170;margin-bottom:20px}
.hero h1 {font-size:2.4rem;margin:0;color:#f3f8ff}
.hero p {color:#b7c9e7;font-size:1rem;margin:8px 0 0}
.smallhint {color:#91a4c4;font-size:0.9rem}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎮 Android Games · ML Studio</h1><p>Hit game classification · Random Forest × XGBoost × Soft Voting</p></div>', unsafe_allow_html=True)

ARTIFACTS = Path(__file__).parent / 'artifacts'
@st.cache_resource
def load_bundle():
    path=ARTIFACTS/'model_bundle.joblib'
    if not path.exists(): return None
    return joblib.load(path)

bundle=load_bundle()
with st.sidebar:
    st.header('⚙️ Workspace')
    page=st.radio('Navigate', ['Overview','Model Comparison','Predict Games','About Project'])
    st.divider()
    st.write('**Model status**')
    if bundle is None:
        st.warning('Models not loaded. Train and export from notebook.')
    else:
        st.success('Trained models loaded')
    st.caption('Only pre-launch candidate features should be used. Verify the target definition for leakage.')

def get_feature_frame(raw):
    cols=bundle['features']
    return raw.reindex(columns=cols)

def score_df(raw):
    X=get_feature_frame(raw)
    rf=bundle['rf'].predict_proba(X)[:,1]
    xgb=bundle['xgb'].predict_proba(X)[:,1]
    w=float(bundle['rf_weight'])
    result=raw.copy()
    result['Random Forest Probability']=rf
    result['XGBoost Probability']=xgb
    result['Voting Probability']=w*rf+(1-w)*xgb
    result['Predicted Hit']=np.where(result['Voting Probability']>=float(bundle['threshold']),'Hit','Not Hit')
    return result

if page=='Overview':
    st.subheader('Project at a glance')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Classification target','is_hit_game')
    c2.metric('Base learners','2')
    c3.metric('Ensemble','Soft Voting')
    c4.metric('Tuning','RandomizedSearchCV')
    st.markdown('### How the prediction pipeline works')
    st.info('Pre-launch features → Missing-value imputation & one-hot encoding → RF / XGBoost → Weighted probability voting → Decision threshold')
    if bundle is None:
        st.warning('**Live results unavailable:** add artifacts/model_bundle.joblib from the trained notebook. No scores or predictions are fabricated.')
    else:
        a,b,c=st.columns(3)
        a.metric('RF weight',f"{bundle['rf_weight']:.0%}")
        b.metric('XGBoost weight',f"{1-bundle['rf_weight']:.0%}")
        c.metric('Voting threshold',f"{bundle['threshold']:.2f}")
    st.markdown('### Workflow')
    st.markdown('**01 — Explore** dataset and class imbalance  →  **02 — Train** RF & XGBoost  →  **03 — Tune** with cross-validation  →  **04 — Compare** holdout results  →  **05 — Predict** new games')
elif page=='Model Comparison':
    st.subheader('Model evaluation')
    if bundle is None or 'metrics' not in bundle:
        st.warning('No real test-set scores available yet. Run the notebook and export the model bundle.')
        st.markdown('**Evaluation metrics:** F1 (Hit) · Recall (Hit) · Precision (Hit) · PR-AUC · ROC-AUC · Accuracy')
    else:
        m=pd.DataFrame(bundle['metrics'])
        st.dataframe(m,hide_index=True,use_container_width=True)
        metrics=[k for k in ['F1 (Hit)','Recall (Hit)','Precision (Hit)','PR-AUC'] if k in m]
        if metrics:
            long=m.melt(id_vars='Model',value_vars=metrics,var_name='Metric',value_name='Score')
            fig=px.bar(long,x='Model',y='Score',color='Metric',barmode='group',template='plotly_dark',title='Holdout performance')
            fig.update_layout(yaxis_range=[0,1])
            st.plotly_chart(fig,use_container_width=True)
        st.caption('Metrics are imported from notebook test-set evaluation; cross-validation results are not test scores.')
elif page=='Predict Games':
    st.subheader('Batch game predictions')
    st.markdown('Upload a CSV with the same input features used for training. Missing features are filled as NaN by the fitted preprocessing pipeline.')
    upload=st.file_uploader('Upload games CSV',type=['csv'])
    if upload is not None:
        df=pd.read_csv(upload)
        st.write('Uploaded records:',len(df))
        if bundle is None: st.error('Please export and add trained models first.')
        else:
            missing=set(bundle['features'])-set(df.columns)
            if missing: st.warning(f'{len(missing)} expected features missing: '+', '.join(sorted(missing)[:15]))
            result=score_df(df)
            st.dataframe(result.head(100),use_container_width=True)
            st.download_button('Download predictions (CSV)',result.to_csv(index=False).encode(),file_name='game_predictions.csv',mime='text/csv')
    elif bundle is not None:
        st.write('**Expected feature columns:**')
        st.code(', '.join(bundle['features']))
    else: st.info('Upload will become available for prediction after the model bundle is added.')
else:
    st.subheader('About this project')
    st.markdown('This project explores a synthetic Android game dataset and benchmarks two tree-based classifiers and a weighted soft-voting ensemble for the binary target `is_hit_game`.')
    st.markdown('**Training:** stratified split · preprocessing inside pipelines · RandomizedSearchCV · PR-AUC selection · out-of-fold weight / threshold search.')
    st.markdown('**Caveats:** class imbalance, synthetic data, potential target leakage and possible optimistic scores from post-hoc model comparison. Do not treat the predicted probability as a calibrated real-world success probability without further validation.')
    st.caption('Made with Streamlit, scikit-learn, XGBoost and Plotly.')
