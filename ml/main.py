import joblib
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

# Charger les données
car_df = pd.read_csv('./data/car-purchase-decision.csv')

# Convertir les variables catégorielles en variables numériques
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(car_df[['Gender']])
X_encoded = pd.DataFrame(X_encoded, columns=['Gender_Male'])

# Concaténer les variables numériques avec le reste du dataframe
X = pd.concat([X_encoded, car_df[['Age', 'AnnualSalary']]], axis=1)
y = car_df['Purchased']  # Renommer la sortie en 'Purchased'

# Entraîner le modèle
model = tree.DecisionTreeClassifier()
model.fit(X, y)

# Enregistrer le modèle
joblib.dump(model, 'car-purchase.joblib')

# Visualiser l'arbre de décision
tree.export_graphviz(
    model,
    out_file='car-purchase.dot',
    feature_names=X.columns,
    class_names=list(map(str, sorted(y.unique()))),  # Convertir les classes en chaînes et utiliser list()
    label='all',
    filled=True
)
