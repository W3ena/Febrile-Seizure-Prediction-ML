# -*- coding: utf-8 -*-
"""lastmycode2.ipynb"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from google.colab import files

# ================== رفع الملف ==================
uploaded = files.upload()
df = pd.read_excel("FS_Dataset_Benghazi_2025.xlsx")

# حذف عمود غير مهم
df = df.drop(columns=["Seizure Type"])

# ================== ترميز الأعمدة النصية ==================
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Family History of Seizures?"] = le.fit_transform(df["Family History of Seizures?"])
df["Did Seizure Occur?"] = le.fit_transform(df["Did Seizure Occur?"])

# ================== معالجة Cause of Fever ==================
df["Cause of Fever"] = df["Cause of Fever"].str.replace("\n", ",").str.strip()
df["Cause of Fever"] = df["Cause of Fever"].str.split(r",\s*")
mlb = MultiLabelBinarizer()
cause_encoded = mlb.fit_transform(df["Cause of Fever"])
cause_df = pd.DataFrame(cause_encoded, columns=mlb.classes_)
df = pd.concat([df.drop("Cause of Fever", axis=1), cause_df], axis=1)

# ================== فصل الميزات والهدف ==================
X = df.drop("Did Seizure Occur?", axis=1)
y = df["Did Seizure Occur?"]

# ================== مجموعات الميزات ==================
cause_features = list(mlb.classes_)

other_features = [
    "Family History of Seizures?",
    "Previous Seizures Count",
    "Fever Duration (days)",
    "Gender",
    "Age (years)"
]

def get_top_features(n):
    selected = cause_features.copy()
    selected.extend(other_features[:n])
    return selected

top3_features = get_top_features(2)
top4_features = get_top_features(3)
top5_features = get_top_features(4)
top6_features = get_top_features(5)

feature_sets = {
    "All Features": X.columns.tolist(),
    "Top 3": top3_features,
    "Top 4": top4_features,
    "Top 5": top5_features,
    "Top 6": top6_features
}

# ================== النماذج ==================
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    ),
    "Support Vector Machine": SVC(
        probability=True,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}

# ================== التدريب والتقييم ==================
for model_name, model in models.items():
    print("=" * 90)
    print(f"🔷 Model: {model_name}")

    for fs_name, features in feature_sets.items():
        X_subset = df[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ===== Pipeline  =====
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        acc = pipeline.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n📌 Feature Set: {fs_name} ")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["No Seizure", "Seizure"]
        ))

        # ===== رسم Confusion Matrix =====
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Seizure", "Seizure"],
            yticklabels=["No Seizure", "Seizure"]
        )
        plt.title(f"{model_name} - {fs_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
