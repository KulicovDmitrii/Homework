# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')
# Установка random_state для воспроизводимости
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# Загрузка данных
df = pd.read_csv('D:/train.csv')

# Просмотр первых строк
df.head()
# Анализ целевой переменной
print("Распределение целевой переменной 'Survived':")
print(df['Survived'].value_counts(normalize=True))
# Выбор признаков
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = df[features]
y = df['Survived']

# Обработка пропущенных значений
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# Кодирование категориальных признаков
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Проверка формы данных
print("Форма X:", X.shape)
print("Количество пропусков:", X.isnull().sum().sum())
# Выбор признаков
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = df[features]
y = df['Survived']

# Обработка пропущенных значений
X['Age'].fillna(X['Age'].median(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# Кодирование категориальных признаков
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Проверка формы данных
print("Форма X:", X.shape)
print("Количество пропусков:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
# Бейзлайн — предсказание наиболее частого класса (в данном случае 0 — погиб)
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)

# Оценка бейзлайна
acc_dummy = accuracy_score(y_test, y_pred_dummy)
f1_dummy = f1_score(y_test, y_pred_dummy)
auc_dummy = roc_auc_score(y_test, y_pred_dummy)

print("Бейзлайн (DummyClassifier):")
print(f"Accuracy: {acc_dummy:.3f}")
print(f"F1-score: {f1_dummy:.3f}")
print(f"ROC-AUC: {auc_dummy:.3f}")
# Модель логистической регрессии
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)

# Обучение на тренировочной выборке
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # вероятности для класса 1
# Вычисление метрик
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("Модель Logistic Regression:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")

# Дополнительно: classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))