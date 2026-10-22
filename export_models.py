"""Execute this file's code INSIDE the notebook after the final comparison cell.
Or run: %run export_models.py from the notebook directory after training.
Requires variables from notebook section 26.
"""
from pathlib import Path
import joblib

required = ['rf_best','xgb_best','best_rf_weight','best_threshold','X_train_ml','results_ml']
not_defined = [key for key in required if key not in globals()]
if not_defined:
    raise RuntimeError('Run notebook training cells first; missing: ' + ', '.join(not_defined))

folder = Path('artifacts')
folder.mkdir(exist_ok=True)
joblib.dump({
    'rf':rf_best,
    'xgb':xgb_best,
    'rf_weight':float(best_rf_weight),
    'threshold':float(best_threshold),
    'features':X_train_ml.columns.tolist(),
    'metrics':results_ml.to_dict(orient='records'),
},folder/'model_bundle.joblib')
print('Saved artifacts/model_bundle.joblib')
