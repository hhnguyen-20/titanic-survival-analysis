import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load data
df = sns.load_dataset('titanic').drop(columns=['pclass', 'embarked', 'alive'])
df.head()

# Format data for dashboard
df.columns = df.columns.str.capitalize().str.replace('_', ' ')
df.rename(columns={'Sex': 'Gender'}, inplace=True)
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.capitalize()

# Partition into train and test splits
TARGET = 'Survived'
y = df[TARGET]
X = df.drop(columns=TARGET)

numerical = X.select_dtypes(include=['number', 'boolean']).columns
categorical = X.select_dtypes(exclude=['number', 'boolean']).columns
X[categorical] = X[categorical].astype('object')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42,
                                                    stratify=y)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OneHotEncoder(sparse_output=False))

        ]), categorical),
        ('num', SimpleImputer(strategy='mean'), numerical)
    ])),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)


# Add predicted probabilities
test['Probability'] = pipeline.predict_proba(X_test)[:,1]
test['Target'] = test[TARGET]
test[TARGET] = test[TARGET].map({0: 'No', 1: 'Yes'})

labels = []
for i, x in enumerate(np.arange(0, 101, 10)):
    if i>0:
        labels.append(f"{previous_x}% to <{x}%")
    previous_x = x
test['Binned probability'] = pd.cut(test['Probability'], len(labels), labels=labels,
                                    right=False)

test
