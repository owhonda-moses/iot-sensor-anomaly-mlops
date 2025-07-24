import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings("ignore")

import math, time, pickle, logging, argparse, joblib, sys
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from imblearn.over_sampling import RandomOverSampler

import optuna
from tqdm.auto import tqdm
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import save_model, Model
from tensorflow.keras.optimizers import Adam


# Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local SQLite database
MLFLOW_EXPERIMENT_NAME = "iot_anomaly_detection"
MODEL_DIR = "../iot_models"
DATA_PATH = "../data/iot_telemetry_data.csv"

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
except mlflow.exceptions.MlflowException:
    pass  # Experiment already exists
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Setup directories
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(path: str) -> pd.DataFrame:
    """Load CSV data with caching"""
    logger = get_run_logger()
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    
    # Log data info to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_metric("unique_devices", df['device'].nunique())
        mlflow.log_metric("date_range_days", (df['ts'].max() - df['ts'].min()).days)
    return df.sort_values('ts').reset_index(drop=True)


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def engineer_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add mappings and flag features"""
    logger = get_run_logger()
    logger.info("Adding feature mappings and flags")
    
    # env/device maps
    env_map = {
        "00:0f:00:70:91:0a": "stable_cooler_humid",
        "1c:bf:ce:15:ec:4d": "variable_temp_humid",
        "b8:27:eb:bf:9d:51": "stable_warmer_dry"
    }
    df['env'] = df['device'].map(env_map)
    df['device'] = df['device'].map({k: f"device_{i+1}" 
                                     for i,k in enumerate(env_map)})

    # timestamp diffs & duplicates
    df['ts_diff'] = df.groupby('device')['ts'].diff().dt.total_seconds().fillna(0)
    df = df.drop_duplicates(['device','ts'])

    # quantile flags per device for temp & humidity
    temp_hum = ['temp','humidity']
    quantiles = [0.01, 0.99]
    qt = {}
    for dev in df['device'].unique():
        sub = df[df['device']==dev]
        lo, hi = sub[temp_hum].quantile(quantiles).values.T
        qt[dev] = dict(temp=(math.floor(lo[0]), math.floor(hi[0])),
                       humidity=(math.floor(lo[1]), math.floor(hi[1])))
    for feat in temp_hum:
        df[f"{feat}_flag"] = 0
        for dev,(low,high) in [(d,qt[d][feat]) for d in qt]:
            mask = (df['device']==dev) & ((df[feat]<low)|(df[feat]>high))
            df.loc[mask, f"{feat}_flag"] = 1

    # gas flags
    gas_thr = {'co':0.01, 'lpg':0.01, 'smoke':0.03}
    for feat,thr in gas_thr.items():
        df[f"{feat}_flag"] = (df[feat] > thr).astype(int)

    # Log feature engineering stats to MLflow
    with mlflow.start_run(nested=True):
        flags = ['temp_flag','humidity_flag','co_flag','lpg_flag','smoke_flag']
        anomaly_rate = df[flags].any(axis=1).mean()
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_param("gas_thresholds", gas_thr)
        mlflow.log_param("quantile_thresholds", quantiles)

    return df


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Preprocess data for modeling"""
    logger = get_run_logger()
    
    df['light']  = df['light'].astype(int)
    df['motion'] = df['motion'].astype(int)

    med = df.groupby('device')['ts_diff'].median().replace(0, np.nan).mean()
    WINDOW = 1 if not np.isfinite(med) else max(1, int(30 / med))
    SENSOR_COLS = ['co','lpg','smoke','temp','humidity']
    for c in SENSOR_COLS:
        grp = df.groupby('device')[c].rolling(WINDOW, min_periods=1)
        df[f'{c}_roll_mean'] = grp.mean().reset_index(level=0, drop=True)
        df[f'{c}_roll_std']  = grp.std().fillna(0).reset_index(level=0, drop=True)

    secs = (df['ts'].dt.hour*3600 +df['ts'].dt.minute*60 + df['ts'].dt.second)
    df['tod_sin'] = np.sin(2*np.pi*secs/86400)
    df['tod_cos'] = np.cos(2*np.pi*secs/86400)

    flags = ['temp_flag','humidity_flag','co_flag','lpg_flag','smoke_flag']
    y = df[flags].any(axis=1).astype(int)

    base_feats    = ['co','humidity','light','motion','lpg','smoke','temp','ts_diff']
    rolling_feats = [f'{c}_{agg}' for c in SENSOR_COLS for agg in ['roll_mean','roll_std']]
    cyclic_feats  = ['tod_sin','tod_cos']
    env_ohe = pd.get_dummies(df['env'],    prefix='', dtype=int)
    dev_ohe = pd.get_dummies(df['device'], prefix='', dtype=int)
    X_df = pd.concat([df[base_feats + rolling_feats + cyclic_feats], env_ohe, dev_ohe], axis=1)

    # pad & reorder
    required = ['stable_cooler_humid','variable_temp_humid','stable_warmer_dry','device_1','device_2','device_3']
    for col in required:
        if col not in X_df.columns:
            X_df[col] = 0
    X_cols = base_feats + rolling_feats + cyclic_feats + required
    X_df = X_df[X_cols]

    if test_size > 0:
        Xtr, Xte, ytr, yte = train_test_split(X_df.values, y.values, test_size=test_size, 
                                              shuffle=False, random_state=random_state)
    else:
        Xtr, ytr = X_df.values, y.values
        Xte, yte = np.empty((0, Xtr.shape[1])), np.empty((0,))

    # scale only if split
    if test_size > 0:
        scaler = MinMaxScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
    else:
        scaler = None

    # Log preprocessing info
    with mlflow.start_run(nested=True):
        mlflow.log_param("rolling_window", WINDOW)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", X_df.shape[1])
        mlflow.log_metric("train_size", len(Xtr))
        mlflow.log_metric("test_size", len(Xte))
        mlflow.log_metric("positive_rate_train", ytr.mean() if len(ytr) > 0 else 0)
        mlflow.log_metric("positive_rate_test", yte.mean() if len(yte) > 0 else 0)

    return scaler, Xtr, Xte, ytr, yte, X_cols


@task
def train_usl(X_train_norm, X_test, y_test, params, model_type="baseline", random_state=42):
    """Train unsupervised learning models with MLflow tracking"""
    logger = get_run_logger()
    
    with mlflow.start_run(nested=True, run_name=f"USL_{model_type}"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "unsupervised")
        mlflow.log_param("variant", model_type)
        
        # Fit IsolationForest on normals
        iso = IsolationForest(
            n_estimators=params["if_n_estimators"],
            max_samples=params["if_max_samples"],
            contamination=params["contamination"],
            random_state=random_state
        ).fit(X_train_norm)

        iso_tr = -iso.decision_function(X_train_norm)
        iso_te = -iso.decision_function(X_test)

        # Build & train AE only if w_ae > 0
        ae_tr = np.zeros_like(iso_tr)
        ae_te = np.zeros_like(iso_te)
        ae_model = None

        if params.get("w_ae", 0) > 0:
            logger.info("Training Autoencoder")
            input_dim = X_train_norm.shape[1]
            enc_dim = params["ae_encoding_dim"]
            lr = params["ae_lr"]

            inp = Input(shape=(input_dim,))
            x = Dense(enc_dim*2, activation="relu")(inp)
            x = Dense(enc_dim, activation="relu")(x)
            x = Dense(enc_dim*2, activation="relu")(x)
            out = Dense(input_dim, activation="sigmoid")(x)

            ae_model = Model(inp, out)
            ae_model.compile(optimizer=Adam(lr), loss="mse")

            # Enable MLflow autologging for Keras
            mlflow.tensorflow.autolog(log_models=False)
            
            history = ae_model.fit(
                X_train_norm, X_train_norm,
                epochs=params["ae_epochs"],
                batch_size=params["ae_batch_size"],
                validation_split=0.1,
                shuffle=True,
                verbose=0
            )

            # Log final training loss
            mlflow.log_metric("ae_final_loss", history.history['loss'][-1])
            if 'val_loss' in history.history:
                mlflow.log_metric("ae_final_val_loss", history.history['val_loss'][-1])

            Xtr_pred = ae_model.predict(X_train_norm, verbose=0)
            ae_tr = np.mean((Xtr_pred - X_train_norm)**2, axis=1)

            Xte_pred = ae_model.predict(X_test, verbose=0)
            ae_te = np.mean((Xte_pred - X_test)**2, axis=1)

        # Stack & scale IF+AE scores into [0,1]
        mat_tr = np.vstack([iso_tr, ae_tr]).T
        mat_te = np.vstack([iso_te, ae_te]).T
        score_scaler = MinMaxScaler().fit(mat_tr)
        norm_tr = score_scaler.transform(mat_tr)
        norm_te = score_scaler.transform(mat_te)

        # Fuse & threshold
        w_if, w_ae, thr_q = params["w_if"], params["w_ae"], params["thr_q"]
        fused_tr = w_if * norm_tr[:,0] + w_ae * norm_tr[:,1]
        fused_te = w_if * norm_te[:,0] + w_ae * norm_te[:,1]
        thr = np.quantile(fused_tr, thr_q)

        # Predict & eval
        y_pred = (fused_te >= thr).astype(int)
        metrics = {
            "roc_auc": round(roc_auc_score(y_test, fused_te), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3)
        }

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        mlflow.log_metric("threshold", thr)
        
        # Log models
        mlflow.sklearn.log_model(iso, "isolation_forest")
        if ae_model is not None:
            mlflow.tensorflow.log_model(ae_model, "autoencoder")

        logger.info(f"USL {model_type} metrics: {metrics}")

        return {
            "iso_model": iso,
            "ae_model": ae_model,
            "score_scaler": score_scaler,
            "threshold": thr,
            "weights": {"if": w_if, "ae": w_ae},
            "fused_scores": fused_te,
            "y_pred": y_pred,
            "metrics": metrics
        }


@task
def train_sl(X_tr, y_tr, X_te, y_te, params, model_type="baseline", random_state=42, n_splits=5):
    """Train supervised learning model with MLflow tracking"""
    logger = get_run_logger()
    
    with mlflow.start_run(nested=True, run_name=f"SL_{model_type}"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "supervised")
        mlflow.log_param("variant", model_type)
        mlflow.log_param("cv_splits", n_splits)
        
        ros = RandomOverSampler(random_state=random_state)
        X_tr_os_all, y_tr_os_all = ros.fit_resample(X_tr, y_tr)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        fold_thresholds = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_tr)):
            mask = np.isin(ros.sample_indices_, train_idx)
            X_tr_os, y_tr_os = X_tr_os_all[mask], y_tr_os_all[mask]
            X_val_f, y_val_f = X_tr[val_idx], y_tr[val_idx]

            model = LogisticRegression(
                C=params["C"], 
                solver="liblinear", 
                class_weight="balanced", 
                max_iter=100, 
                warm_start=True,
                tol=params["tol"], 
                penalty=params["penalty"], 
                n_jobs=-1, 
                random_state=random_state
            )
            model.fit(X_tr_os, y_tr_os)

            # sweep threshold on fold's val
            probs = model.predict_proba(X_val_f)[:,1]
            thresholds = np.linspace(0.10, 0.99, 90)
            f1s = [f1_score(y_val_f, probs>=t) for t in thresholds]
            best_i = int(np.argmax(f1s))
            best_t = thresholds[best_i]

            y_pred = (probs >= best_t).astype(int)
            fold_metric = {
                "roc_auc": round(roc_auc_score(y_val_f, probs), 3),
                "precision": round(precision_score(y_val_f, y_pred, zero_division=0), 3),
                "recall": round(recall_score(y_val_f, y_pred, zero_division=0), 3),
                "f1": round(f1s[best_i], 3)
            }
            fold_metrics.append(fold_metric)
            fold_thresholds.append(best_t)
            
            # Log fold metrics
            for metric_name, value in fold_metric.items():
                mlflow.log_metric(f"fold_{fold}_{metric_name}", value)
            mlflow.log_metric(f"fold_{fold}_threshold", best_t)

        # aggregate
        avg_metrics = {
            k: round(np.mean([m[k] for m in fold_metrics]), 3)
            for k in fold_metrics[0]
        }
        avg_thr = round(float(np.mean(fold_thresholds)), 3)
        
        # Log cross-validation metrics
        for metric_name, value in avg_metrics.items():
            mlflow.log_metric(f"cv_{metric_name}", value)
        mlflow.log_metric("cv_threshold", avg_thr)
        
        # final oversampled refit on full train
        ros_full = RandomOverSampler(random_state=random_state)
        X_full_os, y_full_os = ros_full.fit_resample(X_tr, y_tr)
        logger.info("Fitting final model on oversampled train set")
        final_model = LogisticRegression(
            C=params["C"], 
            solver="liblinear", 
            class_weight="balanced", 
            max_iter=100, 
            warm_start=True, 
            tol=params["tol"], 
            penalty=params["penalty"], 
            n_jobs=-1, 
            random_state=random_state
        )
        final_model.fit(X_full_os, y_full_os)

        # evaluate on test set
        probs_test = final_model.predict_proba(X_te)[:,1]
        y_test_pred = (probs_test >= avg_thr).astype(int)
        test_metrics = {
            "roc_auc": round(roc_auc_score(y_te, probs_test), 3),
            "precision": round(precision_score(y_te, y_test_pred, zero_division=0), 3),
            "recall": round(recall_score(y_te, y_test_pred, zero_division=0), 3),
            "f1": round(f1_score(y_te, y_test_pred), 3)
        }
        
        # Log test metrics
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        
        # Log final model
        mlflow.sklearn.log_model(final_model, "logistic_regression")
        
        logger.info(f"SL {model_type} test metrics: {test_metrics}")

        return {
            "model": final_model,
            "avg_metrics": avg_metrics,
            "avg_thr": avg_thr,
            "test_metrics": test_metrics
        }


@task
def save_artifacts(usl_base: dict, sl_base: dict, sl_base_params: dict, 
                   usl_opt: dict, sl_opt: dict, sl_opt_params: dict, scaler, model_dir: str):
    """Save model artifacts with MLflow tracking"""
    logger = get_run_logger()
    
    for tag, usl_res, sl_res, sl_params in [
        ("baseline", usl_base, sl_base, sl_base_params), 
        ("optimized", usl_opt, sl_opt, sl_opt_params)
    ]:
        out_dir = os.path.join(model_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
        
        # Save IF
        with open(os.path.join(out_dir, "iso.pkl"), "wb") as f:
            pickle.dump(usl_res["iso_model"], f)
        
        # Save Autoencoder if exists
        ae = usl_res.get("ae_model")
        if ae is not None:
            save_model(ae, os.path.join(out_dir, "autoencoder.keras"))
        
        # Save ensemble metadata
        ensemble_meta = {
            "weights": usl_res["weights"], 
            "threshold": float(usl_res["threshold"])
        }
        with open(os.path.join(out_dir, "ensemble.pkl"), "wb") as f:
            pickle.dump(ensemble_meta, f)

        # Save supervised model
        joblib.dump(sl_res["model"], os.path.join(out_dir, "sl_model.pkl"))
        
        # Save SL metadata
        sl_meta = {
            "threshold": sl_res["avg_thr"], 
            "metrics": sl_res["test_metrics"], 
            "params": sl_params
        }
        with open(os.path.join(out_dir, "sl_meta.pkl"), "wb") as f:
            pickle.dump(sl_meta, f)
        
        logger.info(f"{tag.capitalize()} artifacts saved to {out_dir}")


# Model configurations
baseline_usl_params = {
    "if_n_estimators": 100, "if_max_samples": "auto", "contamination": 0.01,
    "ae_encoding_dim": 16, "ae_lr": 1e-3, "ae_epochs": 30, "ae_batch_size": 128,
    "w_if": 1, "w_ae": 0, "thr_q": 0.95
}

optimized_usl_params = {
    'if_n_estimators': 148, 'if_max_samples': 1.0, 'contamination': 0.075,
    'ae_encoding_dim': 8, 'ae_lr': 0.0004, 'ae_epochs': 43, 'ae_batch_size': 64,
    'w_if': 0.658, 'w_ae': 0.342, 'thr_q': 0.969
}

baseline_sl_params = {"C": 1.0, "tol": 1e-4, "penalty": "l2"}
optimized_sl_params = {"C": 5.23, "tol": 0.002, "penalty": "l2"}


@flow(name="IoT Anomaly Detection Pipeline", 
      description="End-to-end pipeline for IoT anomaly detection with USL and SL models")
def main_flow(
    data_path: str = DATA_PATH,
    model_dir: str = MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Main pipeline flow with comprehensive MLflow tracking"""
    
    with mlflow.start_run(run_name="IoT_Pipeline_Run"):
        # Log pipeline parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_dir", model_dir)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        
        start_time = time.time()
        
        # Data loading and preprocessing
        df = load_data(data_path)
        flags_df = engineer_flags(df)
        scaler, Xtr, Xte, ytr, yte, X_cols = preprocess(flags_df, test_size, random_state)
        Xtr_norm = Xtr[ytr == 0]  # Normal samples for USL training

        # Train USL models
        usl_base = train_usl(Xtr_norm, Xte, yte, baseline_usl_params, "baseline", random_state)
        usl_opt = train_usl(Xtr_norm, Xte, yte, optimized_usl_params, "optimized", random_state)

        # Train SL models  
        sl_base = train_sl(Xtr, ytr, Xte, yte, baseline_sl_params, "baseline", random_state)
        sl_opt = train_sl(Xtr, ytr, Xte, yte, optimized_sl_params, "optimized", random_state)

        # Save artifacts
        save_artifacts(usl_base, sl_base, baseline_sl_params,
                      usl_opt, sl_opt, optimized_sl_params, scaler, model_dir)

        # Log overall pipeline metrics
        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60
        mlflow.log_metric("runtime_minutes", runtime_minutes)
        
        # Log model comparison
        mlflow.log_metric("usl_improvement_f1", 
                         usl_opt["metrics"]["f1"] - usl_base["metrics"]["f1"])
        mlflow.log_metric("sl_improvement_f1", 
                         sl_opt["test_metrics"]["f1"] - sl_base["test_metrics"]["f1"])
        
        # Log feature importance if available
        mlflow.log_param("n_features", len(X_cols))
        
        print(f"\n=== PIPELINE RESULTS ===")
        print(f"Runtime: {runtime_minutes:.2f} minutes")
        print(f"\nUSL Baseline F1: {usl_base['metrics']['f1']}")
        print(f"USL Optimized F1: {usl_opt['metrics']['f1']}")
        print(f"USL Improvement: {usl_opt['metrics']['f1'] - usl_base['metrics']['f1']:.3f}")
        print(f"\nSL Baseline F1: {sl_base['test_metrics']['f1']}")
        print(f"SL Optimized F1: {sl_opt['test_metrics']['f1']}")
        print(f"SL Improvement: {sl_opt['test_metrics']['f1'] - sl_base['test_metrics']['f1']:.3f}")


if __name__ == "__main__":
    # Run the pipeline
    main_flow()