"""
Titanic Survival Prediction - Data Science Project
Complete pipeline: Data Cleaning → Visualization → Feature Engineering → Model Building → Evaluation
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'figure.dpi': 150})

# ─────────────────────────────────────────────
# 2. DATA COLLECTION
# ─────────────────────────────────────────────
print("=" * 55)
print("  TITANIC SURVIVAL PREDICTION — DATA SCIENCE PROJECT")
print("=" * 55)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv(url)
    print(f"\n[✓] Dataset loaded from URL — {df.shape[0]} rows × {df.shape[1]} columns")
except Exception:
    # Fallback: synthesize a representative dataset
    print("\n[!] Network unavailable — generating representative Titanic dataset …")
    np.random.seed(42)
    n = 891
    pclass   = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex      = np.random.choice(['male', 'female'], n, p=[0.65, 0.35])
    age      = np.where(np.random.rand(n) < 0.2, np.nan,
                        np.clip(np.random.normal(29, 14, n), 1, 80))
    sibsp    = np.random.choice([0,1,2,3,4], n, p=[0.68,0.23,0.05,0.02,0.02])
    parch    = np.random.choice([0,1,2,3],   n, p=[0.76,0.13,0.08,0.03])
    fare     = np.where(pclass==1,
                        np.abs(np.random.normal(84, 78, n)),
                        np.where(pclass==2,
                                 np.abs(np.random.normal(21, 13, n)),
                                 np.abs(np.random.normal(13, 12, n))))
    survived = np.where(
        sex == 'female',
        np.random.choice([0,1], n, p=[0.26,0.74]),
        np.where(pclass == 1,
                 np.random.choice([0,1], n, p=[0.63,0.37]),
                 np.random.choice([0,1], n, p=[0.82,0.18]))
    )
    embarked_raw = np.random.choice(['S','C','Q',None], n, p=[0.72,0.19,0.086,0.004])
    df = pd.DataFrame({
        'PassengerId': range(1, n+1), 'Survived': survived, 'Pclass': pclass,
        'Name': [f'Passenger_{i}' for i in range(1, n+1)],
        'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch,
        'Ticket': [f'T{i}' for i in range(1, n+1)],
        'Fare': fare, 'Cabin': np.nan,
        'Embarked': embarked_raw
    })
    print(f"[✓] Representative dataset created — {df.shape[0]} rows × {df.shape[1]} columns")

print("\n── Raw Data Preview ──")
print(df.head())
print("\n── Data Info ──")
print(df.info())
print("\n── Missing Values ──")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# 3. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n\n── Step 2: Data Cleaning ──")

df_clean = df.copy()

# Drop Cabin (too many missing), Name, Ticket, PassengerId first
df_clean.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

# Fill missing Age with median
age_median = df_clean['Age'].median()
df_clean['Age'] = df_clean['Age'].fillna(age_median)

# Fill missing Embarked with mode
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

# Encode Sex
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked
df_clean['Embarked'] = df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(f"[✓] Missing values after cleaning:\n{df_clean.isnull().sum()}")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n── Step 3: Feature Engineering ──")

df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone']    = (df_clean['FamilySize'] == 1).astype(int)

print(f"[✓] New features created: FamilySize, IsAlone")
print(df_clean[['SibSp','Parch','FamilySize','IsAlone']].head(8))

# ─────────────────────────────────────────────
# 5. DATA VISUALIZATION
# ─────────────────────────────────────────────
print("\n── Step 4: Data Visualization ──")

COLORS = ['#E74C3C', '#2ECC71']  # red = did not survive, green = survived

# ── Figure 1: Overview (4-panel) ──
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle("Titanic Dataset — Exploratory Data Analysis", fontsize=15, fontweight='bold', y=1.01)

# 4a. Survival Count
survival_counts = df['Survived'].value_counts()
axes[0,0].bar(['Did Not Survive', 'Survived'], survival_counts.values,
               color=COLORS, edgecolor='white', linewidth=1.5, width=0.5)
axes[0,0].set_title("Overall Survival Count")
axes[0,0].set_ylabel("Number of Passengers")
for i, v in enumerate(survival_counts.values):
    axes[0,0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 4b. Survival by Sex
sex_surv = df.groupby(['Sex','Survived']).size().unstack()
sex_surv.index = ['Female', 'Male']
sex_surv.plot(kind='bar', ax=axes[0,1], color=COLORS, edgecolor='white',
              linewidth=1.2, rot=0, legend=True)
axes[0,1].set_title("Survival by Sex")
axes[0,1].set_ylabel("Number of Passengers")
axes[0,1].legend(['Did Not Survive', 'Survived'], loc='upper right')

# 4c. Age Distribution by Survival
for s, color, label in zip([0,1], COLORS, ['Did Not Survive','Survived']):
    axes[1,0].hist(df[df['Survived']==s]['Age'].dropna(), bins=25,
                   alpha=0.65, color=color, label=label, edgecolor='white')
axes[1,0].set_title("Age Distribution by Survival")
axes[1,0].set_xlabel("Age")
axes[1,0].set_ylabel("Count")
axes[1,0].legend()

# 4d. Survival by Pclass
pclass_surv = df.groupby(['Pclass','Survived']).size().unstack()
pclass_surv.index = ['1st Class', '2nd Class', '3rd Class']
pclass_surv.plot(kind='bar', ax=axes[1,1], color=COLORS, edgecolor='white',
                 linewidth=1.2, rot=0, legend=True)
axes[1,1].set_title("Survival by Passenger Class")
axes[1,1].set_ylabel("Number of Passengers")
axes[1,1].legend(['Did Not Survive', 'Survived'])

fig1.tight_layout()
fig1.savefig('/home/claude/plot1_overview.png', bbox_inches='tight', dpi=150)
plt.close()
print("[✓] plot1_overview.png saved")

# ── Figure 2: Fare & Family ──
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Fare Distribution & Family Size Analysis", fontsize=14, fontweight='bold')

# Fare histogram
for s, color, label in zip([0,1], COLORS, ['Did Not Survive','Survived']):
    axes2[0].hist(df[df['Survived']==s]['Fare'].dropna(), bins=30,
                  alpha=0.65, color=color, label=label, edgecolor='white')
axes2[0].set_title("Fare Distribution by Survival")
axes2[0].set_xlabel("Fare (£)")
axes2[0].set_ylabel("Count")
axes2[0].set_xlim(0, 300)
axes2[0].legend()

# Family size
df_clean['FamilySizeGroup'] = pd.cut(df_clean['FamilySize'], bins=[0,1,4,11],
                                      labels=['Alone', 'Small (2–4)', 'Large (5+)'])
fam_surv = df_clean.groupby(['FamilySizeGroup','Survived']).size().unstack()
fam_surv.plot(kind='bar', ax=axes2[1], color=COLORS, edgecolor='white',
              linewidth=1.2, rot=0)
axes2[1].set_title("Survival by Family Size")
axes2[1].set_xlabel("Family Size Group")
axes2[1].set_ylabel("Count")
axes2[1].legend(['Did Not Survive', 'Survived'])

fig2.tight_layout()
fig2.savefig('/home/claude/plot2_fare_family.png', bbox_inches='tight', dpi=150)
plt.close()
print("[✓] plot2_fare_family.png saved")

# ── Figure 3: Correlation Heatmap ──
fig3, ax3 = plt.subplots(figsize=(9, 7))
corr_cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','FamilySize','IsAlone']
corr_matrix = df_clean[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax3, annot_kws={'size': 9})
ax3.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
fig3.tight_layout()
fig3.savefig('/home/claude/plot3_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("[✓] plot3_heatmap.png saved")

# ─────────────────────────────────────────────
# 6. MODEL BUILDING
# ─────────────────────────────────────────────
print("\n── Step 5: Model Building ──")

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
X = df_clean[features]
y = df_clean['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
accuracy = accuracy_score(y_test, y_pred)

print(f"[✓] Logistic Regression trained")
print(f"\n── Accuracy: {accuracy*100:.2f}% ──")
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive','Survived']))

# ── Figure 4: Confusion Matrix ──
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("Model Evaluation", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Did Not Survive', 'Survived'])
disp.plot(ax=axes4[0], colorbar=False, cmap='Blues')
axes4[0].set_title(f"Confusion Matrix  (Accuracy: {accuracy*100:.1f}%)")

# Feature Importance (coefficients)
coefs = pd.Series(np.abs(model.coef_[0]), index=features).sort_values(ascending=True)
colors_bar = ['#3498DB' if c >= coefs.median() else '#AED6F1' for c in coefs.values]
axes4[1].barh(coefs.index, coefs.values, color=colors_bar, edgecolor='white', linewidth=1)
axes4[1].set_title("Feature Importance (|Coefficient|)")
axes4[1].set_xlabel("Absolute Coefficient Value")

fig4.tight_layout()
fig4.savefig('/home/claude/plot4_evaluation.png', bbox_inches='tight', dpi=150)
plt.close()
print("[✓] plot4_evaluation.png saved")

# ─────────────────────────────────────────────
# 7. SAVE RESULTS SUMMARY
# ─────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("  PROJECT COMPLETE")
print("=" * 55)
print(f"  Final Model Accuracy : {accuracy*100:.2f}%")
print(f"  Training Samples     : {len(X_train)}")
print(f"  Test Samples         : {len(X_test)}")
cm_flat = cm.ravel()
print(f"  True Negatives       : {cm_flat[0]}")
print(f"  False Positives      : {cm_flat[1]}")
print(f"  False Negatives      : {cm_flat[2]}")
print(f"  True Positives       : {cm_flat[3]}")
print("=" * 55)
print("\nOutput files:")
print("  titanic_project.py       ← this script")
print("  plot1_overview.png       ← survival overview (4 charts)")
print("  plot2_fare_family.png    ← fare & family size")
print("  plot3_heatmap.png        ← correlation heatmap")
print("  plot4_evaluation.png     ← confusion matrix + feature importance")
print("  Titanic_Report.docx      ← full written report")
