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

# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import save_model, Model
# from tensorflow.keras.optimizers import Adam


# configurations
path = ".../data/iot_telemetry_data.csv"
model_dir = ".../iot_models"
os.makedirs(model_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger()


# Configurations for training
# usl configs
baseline_params = {"if_n_estimators": 100, "if_max_samples": "auto", "contamination": 0.01, # IF
                   "ae_encoding_dim": 16, "ae_lr": 1e-3, "ae_epochs": 30, "ae_batch_size": 128, # AE
                   "w_if": 1, "w_ae": 0, "thr_q": 0.95 # fuse & threshold
}
best_usl = {'if_n_estimators': 148, 'if_max_samples': 1.0, 'contamination': 0.07513614867821206, 
     'ae_encoding_dim': 8, 'ae_lr': 0.0004484570984975867, 'ae_epochs': 43, 'ae_batch_size': 64, 
     'w_if': 0.6582947205466332, 'thr_q': 0.9691031375838758}
best_usl['w_ae'] = 1.0 - best_usl['w_if']
for k,v in baseline_params.items():
    best_usl.setdefault(k, v)

# sl configs
sl_base_params = {"C": 1.0, "tol": 1e-4, "penalty": "l2"}
# best_sl = {'penalty': 'l1', 'C': 11.197186694112919, 'tol': 0.00021993710180958864} #hi
best_sl = {'penalty': 'l2', 'C': 5.232594029557045, 'tol': 0.002263077789430857} #lo

optuna_trials = 100
seed = 42



def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df.sort_values('ts').reset_index(drop=True)



def engineer_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mappings and flag features
    """
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
    df['ts_diff'] = df.groupby('device')['ts'] \
                              .diff().dt.total_seconds().fillna(0)
    # df['ts_large'] = (df['ts_diff'] > 4).astype(int)
    # df['ts_duplicate']= df.duplicated(['device','ts'], keep=False).astype(int)
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

    return df



def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, 
               reference_columns=None, fit_scaler=True):
    """
    Args:
        df: Input dataframe
        test_size: Test split ratio
        random_state: Random seed
        reference_columns: List of column names to match (for prediction)
        fit_scaler: Whether to fit a new scaler (True for training, False for prediction)
    
    Returns:
        scaler (or None if test_size=0),
        Xtr, Xte, ytr, yte, X_cols
    """
    df = df.copy()  # Avoid modifying original df
    df['light']  = df['light'].astype(int)
    df['motion'] = df['motion'].astype(int)
    
    # Calculate window size
    med = df.groupby('device')['ts_diff'].median().replace(0, np.nan).mean()
    WINDOW = 1 if not np.isfinite(med) else max(1, int(30 / med))
    
    # Rolling features
    SENSOR_COLS = ['co','lpg','smoke','temp','humidity']
    for c in SENSOR_COLS:
        grp = df.groupby('device')[c].rolling(WINDOW, min_periods=1)
        df[f'{c}_roll_mean'] = grp.mean().reset_index(level=0, drop=True)
        df[f'{c}_roll_std']  = grp.std().fillna(0).reset_index(level=0, drop=True)
    
    # Time-of-day features
    secs = (df['ts'].dt.hour*3600 + df['ts'].dt.minute*60 + df['ts'].dt.second)
    df['tod_sin'] = np.sin(2*np.pi*secs/86400)
    df['tod_cos'] = np.cos(2*np.pi*secs/86400)
    
    # Target variable
    flags = ['temp_flag','humidity_flag','co_flag','lpg_flag','smoke_flag']
    y = df[flags].any(axis=1).astype(int)
    
    # Base features
    base_feats    = ['co','humidity','light','motion','lpg','smoke','temp','ts_diff']
    rolling_feats = [f'{c}_{agg}' for c in SENSOR_COLS for agg in ['roll_mean','roll_std']]
    cyclic_feats  = ['tod_sin','tod_cos']
    
    # One-hot encoding
    env_ohe = pd.get_dummies(df['env'], prefix='', dtype=int)
    dev_ohe = pd.get_dummies(df['device'], prefix='', dtype=int)
    
    # Combine all features
    X_df = pd.concat([df[base_feats + rolling_feats + cyclic_feats], env_ohe, dev_ohe], axis=1)
    
    # Handle feature consistency for prediction
    if reference_columns is not None:
        # Ensure we have all reference columns
        for col in reference_columns:
            if col not in X_df.columns:
                X_df[col] = 0  # Add missing columns with zeros
        # Reorder and select only reference columns
        X_df = X_df[reference_columns]
        X_cols = reference_columns
    else:
        X_cols = X_df.columns.to_list()
    
    # Train/test split
    if test_size > 0:
        Xtr, Xte, ytr, yte = train_test_split(X_df.values, y.values, test_size=test_size, 
                                              shuffle=False, random_state=random_state)
    else:
        Xtr, ytr = X_df.values, y.values
        Xte, yte = np.empty((0, Xtr.shape[1])), np.empty((0,))
    
    # Scaling
    if test_size > 0 and fit_scaler:
        scaler = MinMaxScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
    else:
        scaler = None
    
    return scaler, Xtr, Xte, ytr, yte, X_cols



def train_usl(X_train_norm, X_test, y_test, params, random_state=42):
    """
    Runs IF + (optional) AE, fuses their anomaly scores, 
    thresholds, and evaluates on y_test.
    """
    # Fit IsolationForest on normals
    w_if = params.get("w_if", 1.0)
    w_ae = 1.0 - w_if

    iso = IsolationForest(
        n_estimators = params["if_n_estimators"],
        max_samples = params["if_max_samples"],
        contamination = params["contamination"],
        random_state = random_state
    ).fit(X_train_norm)

    iso_tr = -iso.decision_function(X_train_norm)
    iso_te = -iso.decision_function(X_test)

    # Initialize autoencoder variables
    ae_tr = np.zeros_like(iso_tr)
    ae_te = np.zeros_like(iso_te)
    ae_model = None

    # hashed out for CPU execution
    # if params.get("w_ae", 0) > 0:
    #     input_dim = X_train_norm.shape[1]
    #     enc_dim =  params["ae_encoding_dim"]
    #     lr = params["ae_lr"]

    #     inp = Input(shape=(input_dim,))
    #     x   = Dense(enc_dim*2, activation="relu")(inp)
    #     x   = Dense(enc_dim,   activation="relu")(x)
    #     x   = Dense(enc_dim*2, activation="relu")(x)
    #     out = Dense(input_dim,  activation="sigmoid")(x)

    #     ae_model = Model(inp, out)
    #     ae_model.compile(optimizer=Adam(lr), loss="mse")

    #     ae_model.fit(
    #         X_train_norm, X_train_norm,
    #         epochs = params["ae_epochs"],
    #         batch_size = params["ae_batch_size"],
    #         validation_split = 0.1,
    #         shuffle = True,
    #         verbose = 0
    #     )

    #     Xtr_pred = ae_model.predict(X_train_norm, verbose=0)
    #     ae_tr    = np.mean((Xtr_pred - X_train_norm)**2, axis=1)

    #     Xte_pred = ae_model.predict(X_test, verbose=0)
    #     ae_te    = np.mean((Xte_pred - X_test)**2, axis=1)

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




def train_sl(X_tr, y_tr, X_te, y_te, C, tol, penalty, random_state, n_splits=5):
    """
    TimeSeriesSplit CV + threshold tuning + final refit.

    Inputs:
      X_tr, y_tr      — full train arrays
      C               — LR regularization param
      random_state    — for sampler & LR
      n_splits        — number of TS folds

    Prints per-fold thr & metrics, returns:
      final_model, avg_metrics_dict, avg_thr
    """
    ros = RandomOverSampler(random_state=random_state)
    X_tr_os_all, y_tr_os_all = ros.fit_resample(X_tr, y_tr)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics    = []
    fold_thresholds = []
    
    for train_idx, val_idx in tscv.split(X_tr):
        mask = np.isin(ros.sample_indices_, train_idx)
        X_tr_os, y_tr_os = X_tr_os_all[mask], y_tr_os_all[mask]
        X_val_f, y_val_f = X_tr[val_idx], y_tr[val_idx]

        model = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=100, warm_start=True,
                                   tol=tol, penalty=penalty, n_jobs=-1, random_state=random_state)
        # model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=1, random_state=random_state)
        model.fit(X_tr_os, y_tr_os)

        # sweep threshold on fold’s val
        probs  = model.predict_proba(X_val_f)[:,1]
        thresholds = np.linspace(0.10, 0.99, 90)
        f05s = [fbeta_score(y_val_f, probs>=t, beta=0.5) for t in thresholds]
        f1s = [f1_score(y_val_f, probs>=t) for t in thresholds]
        best_i = int(np.argmax(f1s))
        best_t = thresholds[best_i]
        best_f05 = f05s[best_i]

        y_pred = (probs >= best_t).astype(int)
        metrics = {
            "roc_auc":   round(roc_auc_score(y_val_f, probs), 3),
            "precision": round(precision_score(y_val_f, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_val_f, y_pred, zero_division=0), 3),
            "f1":        round(f1s[best_i], 3),
            "f05":       round(best_f05, 3)
        }
        fold_metrics.append(metrics)
        fold_thresholds.append(best_t)
        # logger.info(f"Fold {fold}: thr={best_t:.3f}, f1={metrics['f1']:.3f}")

    # aggregate
    avg_metrics = {
        k: round(np.mean([m[k] for m in fold_metrics]), 3)
        for k in fold_metrics[0]
    }
    avg_thr = round(float(np.mean(fold_thresholds)), 3)
    
    # final oversampled refit on full train
    ros_full  = RandomOverSampler(random_state=random_state)
    X_full_os, y_full_os = ros_full.fit_resample(X_tr, y_tr)
    logger.info("Fitting model on oversampled train set")
    final_model = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=100, warm_start=True, 
                                     tol=tol, penalty=penalty, n_jobs=-1, random_state=random_state)
    # final_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=1, random_state=random_state)
    final_model.fit(X_full_os, y_full_os)

    # evaluate on test set
    probs_test = final_model.predict_proba(X_te)[:,1]
    y_test_pred = (probs_test >= avg_thr).astype(int)
    test_metrics = {
        "roc_auc":   round(roc_auc_score(y_te, probs_test), 3),
        "precision": round(precision_score(y_te, y_test_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_te, y_test_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_te, y_test_pred), 3),
        "f05":      round(fbeta_score(y_te, y_test_pred, beta=0.5), 3)
    }
    # logger.info(f"CV metrics: {avg_metrics} ‖ thr: {avg_thr}")
    # logger.info(f"SL metrics: {test_metrics}")


    return {
        "model":        final_model,
        "avg_metrics":  avg_metrics,
        "avg_thr":      avg_thr,
        "test_metrics": test_metrics
    }



def save_artifacts(usl_base: dict, sl_base: dict, sl_base_params: dict, 
                   usl_opt: dict, sl_opt: dict, sl_opt_params: dict, scaler, X_cols, model_dir: str = "iot_models"):
    """
    Persist baseline and optimized USL+SL artifacts under:
      model_dir/baseline/...  
      model_dir/optimized/...
    """
    for tag, usl_res, sl_res, sl_params in [
        ("baseline", usl_base, sl_base, sl_base_params), ("optimized", usl_opt,  sl_opt,  sl_opt_params)
        ]:
        out_dir = os.path.join(model_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl")) # scaler
        with open(os.path.join(out_dir, "X_cols.pkl"), "wb") as f: #X_cols
            pickle.dump(X_cols, f)
        with open(os.path.join(out_dir, "iso.pkl"), "wb") as f: # IF
            pickle.dump(usl_res["iso_model"], f)
        ae = usl_res.get("ae_model") # Autoencoder
        if ae is not None:
            save_model(ae, os.path.join(out_dir, "autoencoder.keras"))
        ensemble_meta = {"weights": usl_res["weights"], "threshold": float(usl_res["threshold"])} # ensemble metadata
        with open(os.path.join(out_dir, "ensemble.pkl"), "wb") as f:
            pickle.dump(ensemble_meta, f)

        # supervised model
        joblib.dump(sl_res["model"], os.path.join(out_dir, "sl_model.pkl"))
        sl_meta = {"threshold": sl_res["avg_thr"], "metrics": sl_res["test_metrics"], "params": sl_params} # SL metadata
        with open(os.path.join(out_dir, "sl_meta.pkl"), "wb") as f:
            pickle.dump(sl_meta, f)
        logger.info(f"{tag.capitalize()} artifacts saved to {out_dir}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_optuna_usl", action="store_true")
    parser.add_argument("--run_optuna_sl", action="store_true")
    args, _ = parser.parse_known_args()

    start = time.time()
    # load & preprocess
    df = load_data(path)
    df = engineer_flags(df)
    scaler, Xtr, Xte, ytr, yte, X_cols = preprocess(df)
    Xtr_norm = Xtr[ytr == 0]  

    # USL
    logger.info("== Baseline USL ==")
    usl_base = train_usl(Xtr_norm, Xte, yte, baseline_params)
    logger.info("Baseline USL metrics: %s", usl_base["metrics"])
    if args.run_optuna_usl: # USL tuning
        logger.info("Starting USL tuning with %d trials", optuna_trials)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        pbar = tqdm(total=optuna_trials, desc="USL tuning")
        for _ in range(optuna_trials):
            study.optimize(objective, n_trials=1)
            pbar.update(1)
            pbar.set_postfix(best_f1=f"{study.best_value:.3f}")
        pbar.close()
        usl_best = study.best_trial.params
        for k, v in baseline_params.items(): # back-fill defaults
            usl_best.setdefault(k, v)
        logger.info("Optimized USL params: %s", usl_best)
    else:
        usl_best = best_usl
        logger.info("Skipping USL tuning; using best_usl: %s", usl_best)
    # USL final
    logger.info("== Optimized USL ==")
    usl_opt = train_usl(Xtr_norm, Xte, yte, usl_best)
    logger.info("Optimized USL metrics: %s", usl_opt["metrics"])

    # SL
    logger.info("== Baseline SL ==")
    sl_base = train_sl(X_tr=Xtr, y_tr=ytr, X_te=Xte, y_te=yte,
                        **sl_base_params, random_state=seed, n_splits=5)
    logger.info("Baseline SL test metrics: %s", sl_base["test_metrics"])
    if args.run_optuna_sl: # SL tuning
        logger.info("Starting SL tuning with %d trials", optuna_trials)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        pbar = tqdm(total=optuna_trials, desc="SL tuning")
        study.optimize(objective, n_trials=optuna_trials, n_jobs=4,
                       callbacks=[lambda s, t: (pbar.update(1), pbar.set_postfix(best_f1=f"{s.best_value:.4f}"))])
        pbar.close()
        sl_best = study.best_trial.params
        logger.info("Optimized SL params: %s", sl_best)
    else:
        sl_best = best_sl
        logger.info("Skipping SL tuning; using best_sl: %s", sl_best)
    # SL final
    logger.info("== SL OPTIMIZED ==")
    sl_opt = train_sl(X_tr=Xtr,y_tr=ytr, X_te=Xte, y_te=yte,
            **sl_best, random_state=seed, n_splits=5)
    logger.info("Optimized SL test metrics: %s", sl_opt["test_metrics"])

    # save artifacts
    save_artifacts(usl_base=usl_base, sl_base=sl_base, sl_base_params=sl_base_params,
                   usl_opt=usl_opt, sl_opt=sl_opt, sl_opt_params=sl_best, scaler=scaler,
                   X_cols=X_cols, model_dir=model_dir
                   )
    end = time.time()
    logger.info("Runtime: %.2f minutes", (end - start) / 60)




def main_flow(
    path: str  = "./data/iot_telemetry_data.csv",
    tune_usl: bool = False,
    tune_sl:  bool = False,
    model_dir: str = "ensemble_models"
):
    # 1) load & engineer
    df = load_data(path)
    flags_df  = engineer_flags_task(df)

    # 2) preprocess & split
    scaler, Xtr, Xte, ytr, X_cols, _ = preprocess_task(flags_df).result()
    Xtr_norm = Xtr[ytr == 0]

    # 3) USL baseline
    usl_base = train_usl_task(Xtr_norm, Xte, yte, baseline_params)

    # 4) USL optimized (skipping Optuna here for brevity)
    usl_opt  = train_usl_task(Xtr_norm, Xte, yte, best_usl)

    # 5) SL baseline & optimized
    sl_base = train_sl_task(Xtr, ytr, Xte, yte, {"C":1.0, "tol":1e-4, "penalty":"l2"})
    sl_opt  = train_sl_task(Xtr, ytr, Xte, yte, best_sl)

    # 6) save everything
    save_artifacts_task(
      usl_base, sl_base, {"C":1.0,"tol":1e-4,"penalty":"l2"},
      usl_opt, sl_opt,  best_sl,
      scaler, X_cols, model_dir
    )




if __name__ == "__main__":
    main()
    # main_flow(
    #   path="./data/iot_telemetry_data.csv",
    #   tune_usl=False,
    #   tune_sl=False,
    #   model_dir="iot_models"
    # )
