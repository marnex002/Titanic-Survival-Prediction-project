import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data Collection
titanic_data = pd.read_csv('titanic.csv')

# Data Cleaning
# Fill missing values
# Age
mean_age = titanic_data['Age'].mean()
titanic_data['Age'].fillna(mean_age, inplace=True)
# Embarked
mode_embarked = titanic_data['Embarked'].mode()[0]
titanic_data['Embarked'].fillna(mode_embarked, inplace=True)
# Drop unnecessary columns
titanic_data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

# Feature Engineering
# Convert categorical variables to numeric
titanic_data['Sex'] = np.where(titanic_data['Sex'] == 'male', 1, 0)
# One hot encoding for Embarked
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

# Visualization
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

# Split the dataset into training and testing sets
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)