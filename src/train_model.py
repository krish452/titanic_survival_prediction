"""
Titanic Survival Prediction
- Feature engineering from raw data
- Logistic Regression baseline
- ROC-AUC evaluation
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

df = pd.read_csv(r"C:\Users\DELL\Documents\PythonDSA\Titanic\dataset\Titanic-Dataset.csv")
y = df['Survived']
x = df.drop(columns=['Survived', 'PassengerId'])

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=42, stratify=y)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.value_counts(normalize=True),y_test.value_counts(normalize=True))

def feature_engineering(df):
    df['title'] = (
        df['Name']
        .str.split(',', expand=True)[1]
        .str.split('.', expand=True)[0]
        .str.strip()
    )
    common_titles=['Mr', 'Mrs', 'Miss', 'Master']
    df['title'] = df['title'].where(
        df['title'].isin(common_titles), 'Rare'
    )
    age_group_median = df.groupby('title')['Age'].median()
    df['Age'] = df['Age'].fillna(
        df['title'].map(age_group_median)
    )
    df['FamilySize'] = df['SibSp']+df['Parch']+1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = (df['Cabin'].isna()).astype(int)
    df = df.drop(columns=['Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'])
    return df


x_train = feature_engineering(x_train)
x_test = feature_engineering(x_test)

x_train['Sex'] = x_train['Sex'].map({'male':0, 'female':1})
x_test['Sex'] = x_test['Sex'].map({'male':0, 'female':1})

x_train = pd.get_dummies(
    x_train,
    columns=['Embarked', 'title'],
    drop_first=False
)
x_test = pd.get_dummies(
    x_test,
    columns=['Embarked', 'title'],
    drop_first=False
)

scaler = StandardScaler()
scaler.fit(x_train[['Age','Fare']])

x_train[['Age','Fare']] = scaler.transform(x_train[['Age','Fare']])
x_test[['Age', 'Fare']] = scaler.transform(x_test[['Age', 'Fare']])

model = LogisticRegression(random_state=0,max_iter=1000)
model.fit(x_train, y_train)
y_pred_prob = model.predict_proba(x_test)[:,1]
roc_auc = roc_auc_score(y_test,y_pred_prob)
print(roc_auc)
