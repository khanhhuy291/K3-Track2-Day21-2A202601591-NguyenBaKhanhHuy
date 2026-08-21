import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EVAL_THRESHOLD = 0.70


def check_data_distribution(y_train: pd.Series) -> dict:
    """
    Kiem tra phan phoi nhan trong tap huan luyen (Bonus 5).
    In canh bao neu co lop < 10%.
    """
    total = len(y_train)
    dist = (y_train.value_counts() / total).to_dict()
    print("--- Thong ke phan phoi nhan ---")
    for cls, ratio in sorted(dist.items()):
        print(f"Lop {cls}: {ratio * 100:.2f}%")
        if ratio < 0.10:
            print(f"[CANH BAO] Lop {cls} chiem duoi 10% ({ratio * 100:.2f}%) tong mau!")
    return {str(k): float(v) for k, v in dist.items()}


def get_model(params: dict):
    """
    Khoi tao mo hinh dua tren model_type (Bonus 2).
    """
    model_params = params.copy()
    model_type = model_params.pop("model_type", "random_forest")

    if model_type == "gradient_boosting":
        valid_keys = {"n_estimators", "learning_rate", "max_depth", "min_samples_split"}
        filtered = {k: v for k, v in model_params.items() if k in valid_keys and v is not None}
        return GradientBoostingClassifier(**filtered, random_state=42)
    elif model_type == "logistic_regression":
        valid_keys = {"C", "max_iter", "solver"}
        filtered = {k: v for k, v in model_params.items() if k in valid_keys and v is not None}
        return LogisticRegression(**filtered, max_iter=1000, random_state=42)
    else:
        # Default: RandomForest
        return RandomForestClassifier(**model_params, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """
    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 3. Kiem tra phan phoi du lieu (Bonus 5)
    class_dist = check_data_distribution(y_train)

    with mlflow.start_run():
        # 4. Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # 5. Khoi tao va huan luyen mo hinh (Bonus 2: Multi-algorithm)
        model = get_model(params)
        model.fit(X_train, y_train)

        # 6. Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # 7. Tao bao cao chi tiet (Bonus 3: Classification Report & Confusion Matrix)
        clf_report = classification_report(y_eval, preds, zero_division=0)
        conf_matrix = confusion_matrix(y_eval, preds)

        # In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print("\n--- Confusion Matrix ---")
        print(conf_matrix)
        print("\n--- Classification Report ---")
        print(clf_report)

        # 8. Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # 9. Luu metrics & data distribution ra outputs/metrics.json (Bonus 5)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc,
                "f1_score": f1,
                "class_distribution": class_dist
            }, f, indent=2)

        # 10. Luu bao cao chi tiet ra outputs/report.txt (Bonus 3)
        with open("outputs/report.txt", "w") as f:
            f.write(f"MODEL PERFORMANCE REPORT\n{'='*40}\n")
            f.write(f"Accuracy: {acc:.4f}\nF1-score (weighted): {f1:.4f}\n\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
            f.write(f"Classification Report:\n{clf_report}\n")

        # 11. Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
