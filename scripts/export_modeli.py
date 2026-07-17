"""Trenira finalne modele (isti pipeline kao u koji_model_biramo notebookovima) iz cachea
znacajki i exporta ih u ONNX format u mapu modeli_exports/.

Modeli:
  1. Decision Tree s ogranicenjem dubine (max_depth=10)  - decision_tree.ipynb, prvi model
  2. Random Forest (n_estimators=300)                    - random_forest.ipynb
  3. XGBoost s early stoppingom                          - xgboost_model.ipynb, drugi model

Pokretanje:  python scripts/export_modeli.py  (iz roota repoa)
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMlFloatTensorType
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE, 'data', 'znacajke_cache.pkl')
EXPORT_DIR = os.path.join(BASE, 'modeli_exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Podaci - identican split kao u sva tri notebooka
# ------------------------------------------------------------------
df = pd.read_pickle(CACHE_PATH)
X = df.drop(columns=['label'])
y = df['label']
n_features = X.shape[1]
print(f"Ucitano {len(df)} uzoraka, {n_features} znacajki")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


def verify_onnx(path, model, X_check, expected_labels):
    """Provjeri da ONNX model daje iste predikcije kao originalni model."""
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_labels = sess.run(None, {input_name: X_check.to_numpy(dtype=np.float32)})[0]
    match = np.mean(np.asarray(onnx_labels) == np.asarray(expected_labels))
    size_kb = os.path.getsize(path) / 1024
    print(f"  -> {os.path.basename(path)} ({size_kb:.0f} KB), "
          f"podudaranje predikcija ONNX vs original: {match:.4%}")
    return match


# ------------------------------------------------------------------
# 1. Decision Tree (max_depth=10) - "prvi" model iz decision_tree.ipynb
# ------------------------------------------------------------------
print("\n[1/3] Decision Tree (max_depth=10)...")
dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
print(f"  Test accuracy: {dt.score(X_test, y_test):.3f}")

dt_onnx = convert_sklearn(
    dt,
    initial_types=[('input', FloatTensorType([None, n_features]))],
    options={'zipmap': False},  # probabilities kao obican tensor, jednostavnije za kontroler
)
dt_path = os.path.join(EXPORT_DIR, 'decision_tree_max_depth10.onnx')
with open(dt_path, 'wb') as f:
    f.write(dt_onnx.SerializeToString())
verify_onnx(dt_path, dt, X_test, dt.predict(X_test))

# ------------------------------------------------------------------
# 2. Random Forest (n_estimators=300) - random_forest.ipynb
# ------------------------------------------------------------------
print("\n[2/3] Random Forest (n_estimators=300)...")
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"  Test accuracy: {rf.score(X_test, y_test):.3f}")

rf_onnx = convert_sklearn(
    rf,
    initial_types=[('input', FloatTensorType([None, n_features]))],
    options={'zipmap': False},
)
rf_path = os.path.join(EXPORT_DIR, 'random_forest_300.onnx')
with open(rf_path, 'wb') as f:
    f.write(rf_onnx.SerializeToString())
verify_onnx(rf_path, rf, X_test, rf.predict(X_test))

# ------------------------------------------------------------------
# 3. XGBoost s early stoppingom - "drugi" model iz xgboost_model.ipynb
# ------------------------------------------------------------------
print("\n[3/3] XGBoost (early stopping)...")
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
print(f"  Early stopping: najbolja iteracija = {n_best} stabala")

# Refit s tocno n_best stabala bez early stoppinga: boosting je deterministican pa su
# stabla identicna, a exportani model nema "visak" stabala iza best_iteration.
xgb_final = XGBClassifier(n_estimators=n_best, **xgb_params)
xgb_final.fit(X_tr2_np, y_tr2, sample_weight=weights_tr2)
print(f"  Test accuracy: {xgb_final.score(X_test_np, y_test_enc):.3f}")

xgb_onnx = convert_xgboost(
    xgb_final,
    initial_types=[('input', OnnxMlFloatTensorType([None, n_features]))],
)
xgb_path = os.path.join(EXPORT_DIR, 'xgboost_early_stopping.onnx')
with open(xgb_path, 'wb') as f:
    f.write(xgb_onnx.SerializeToString())
verify_onnx(xgb_path, xgb_final, X_test, xgb_final.predict(X_test_np))

# XGBoost radi s numerickim labelama - spremi mapiranje broj -> klasa za kontroler
mapping_path = os.path.join(EXPORT_DIR, 'xgboost_label_mapping.txt')
with open(mapping_path, 'w') as f:
    for i, cls in enumerate(le.classes_):
        f.write(f"{i} {cls}\n")

# Redoslijed znacajki na ulazu (isti za sva tri modela) - kontroler ih mora slati tim redom
features_path = os.path.join(EXPORT_DIR, 'redoslijed_znacajki.txt')
with open(features_path, 'w') as f:
    for i, col in enumerate(X.columns):
        f.write(f"{i} {col}\n")

print(f"\nGotovo! Sve spremljeno u: {EXPORT_DIR}")
for name in sorted(os.listdir(EXPORT_DIR)):
    print(f"  - {name}")
