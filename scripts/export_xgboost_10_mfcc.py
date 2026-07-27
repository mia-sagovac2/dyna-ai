"""Trenira XGBoost (early stopping) na 10-znacajki MFCC cacheu i exporta ga u ONNX,
isti pristup kao scripts/export_modeli.py (samo za ovu jednu varijantu modela).

Pokretanje:  python scripts/export_xgboost_10_mfcc.py  (iz roota repoa)
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMlFloatTensorType
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, 'data', 'features_cache', 'znacajke_cache_10.pkl')
EXPORT_DIR = os.path.join(BASE, 'modeli_exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_pickle(CACHE_PATH)
X = df.drop(columns=['label'])
y = df['label']
n_features = X.shape[1]
print(f"Ucitano {len(df)} uzoraka, {n_features} znacajki")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42
)
weights_tr2 = compute_sample_weight('balanced', y_tr2)

xgb_params = dict(
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)
# ONNX converter za XGBoost zahtijeva featuree bez imena (pattern f0, f1, ...), pa
# treniramo na numpy arrayima umjesto DataFramea - podaci su identicni.
X_tr2_np, X_val_np, X_test_np = X_tr2.to_numpy(), X_val.to_numpy(), X_test.to_numpy()

clf_es = XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **xgb_params)
clf_es.fit(X_tr2_np, y_tr2, sample_weight=weights_tr2,
           eval_set=[(X_val_np, y_val)], verbose=False)
n_best = clf_es.best_iteration + 1
print(f"Early stopping: najbolja iteracija = {n_best} stabala")

# Refit s tocno n_best stabala bez early stoppinga: boosting je deterministican pa su
# stabla identicna, a exportani model nema "visak" stabala iza best_iteration.
xgb_final = XGBClassifier(n_estimators=n_best, **xgb_params)
xgb_final.fit(X_tr2_np, y_tr2, sample_weight=weights_tr2)
print(f"Test accuracy: {xgb_final.score(X_test_np, y_test_enc):.3f}")

xgb_onnx = convert_xgboost(
    xgb_final,
    initial_types=[('input', OnnxMlFloatTensorType([None, n_features]))],
)
xgb_path = os.path.join(EXPORT_DIR, 'xgboost_10_mfcc.onnx')
with open(xgb_path, 'wb') as f:
    f.write(xgb_onnx.SerializeToString())

sess = ort.InferenceSession(xgb_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
onnx_preds = sess.run(None, {input_name: X_test_np.astype(np.float32)})[0]
sklearn_preds = xgb_final.predict(X_test_np)
match = np.mean(np.asarray(onnx_preds) == np.asarray(sklearn_preds))
size_kb = os.path.getsize(xgb_path) / 1024
print(f"  -> {os.path.basename(xgb_path)} ({size_kb:.0f} KB), "
      f"podudaranje predikcija ONNX vs original: {match:.4%}")

# XGBoost radi s numerickim labelama - spremi mapiranje broj -> klasa za kontroler
mapping_path = os.path.join(EXPORT_DIR, 'xgboost_10_mfcc_label_mapping.txt')
with open(mapping_path, 'w') as f:
    for i, cls in enumerate(le.classes_):
        f.write(f"{i} {cls}\n")

# Redoslijed znacajki na ulazu - kontroler ih mora slati tim redom
features_path = os.path.join(EXPORT_DIR, 'xgboost_10_mfcc_redoslijed_znacajki.txt')
with open(features_path, 'w') as f:
    for i, col in enumerate(X.columns):
        f.write(f"{i} {col}\n")

print(f"\nGotovo! Spremljeno u: {EXPORT_DIR}")
