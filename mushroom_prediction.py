import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import chi2_contingency


mushroom_data = pd.read_csv("./mushrooms.csv")
f = open("./mushrooms_encoded_description.txt", "r")
mushroom_description = f.readlines()
f.close()

mushroom_decoded_traits = []

for trait in mushroom_description:
    traits = re.findall(r"[a-z]+=", trait)
    letters = re.findall(r"=[a-z?]*", trait)
    decoded_traits_dict = {}
    for i in range(0, len(traits)):
        decoded_traits_dict[letters[i][1]] = traits[i][:-1]
    mushroom_decoded_traits.append(decoded_traits_dict)


#print(mushroom_data.head())
#print(mushroom_data.describe())
#print(mushroom_data.isna().sum())
col = mushroom_data.columns

for i in range(0, len(col)):
    mushroom_data[col[i]] = mushroom_data[col[i]].replace(mushroom_decoded_traits[i])

#print(mushroom_data.head(5))
mushrooms_data_encoded = pd.get_dummies(mushroom_data, columns=list(mushroom_data.columns[1:]))
#print(mushrooms_data_encoded.head())

le = LabelEncoder()

mushrooms_data_encoded["class"] = le.fit_transform(mushrooms_data_encoded["class"])
#edible: 0, poisonous: 1
#print(mushrooms_data_encoded.head())

X = mushrooms_data_encoded.drop("class", axis=1)
y = mushrooms_data_encoded["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_mushrooms = RandomForestClassifier()
model_mushrooms.fit(X_train, y_train)

prediction = model_mushrooms.predict(X_test)
print(prediction[:5])
print(y_test.head())

print("Dokładność:", accuracy_score(y_test, prediction))
print("\nRaport klasyfikacji:\n", classification_report(y_test, prediction))
print("\nMacierz pomyłek:\n", confusion_matrix(y_test, prediction))
test = {
    'cap-shape': ["convex", "flat"], 
    'cap-surface': ["smooth", "scaly"], 
    'cap-color': ["white", "brown"], 
    'bruises': ["bruises", "no"], 
    'odor': ["almond", "none"],
    'gill-attachment': ["free", "free"], 
    'gill-spacing': ["close", "close"], 
    'gill-size': ["broad", "broad"], 
    'gill-color': ["brown", "white"],
    'stalk-shape': ["enlarging", "enlarging"], 
    'stalk-root': ["bulbous", "bulbous"], 
    'stalk-surface-above-ring': ["smooth", "scaly"],
    'stalk-surface-below-ring': ["smooth", "scaly"], 
    'stalk-color-above-ring': ["white", "brown"],
    'stalk-color-below-ring': ["white", "brown"], 
    'veil-type': ["partial", "partial"], 
    'veil-color': ["white", "white"], 
    'ring-number': ["one", "one"],
    'ring-type': ["pendant", "pendant"], 
    'spore-print-color': ["brown", "white"], 
    'population': ["numerous", "scattered"], 
    'habitat': ["urban", "meadows"]
}
sample = {
    'cap-shape': 'convex',
    'cap-surface': 'smooth',
    'cap-color': 'white',
    'bruises': 'bruises',
    'odor': 'almond',
    'gill-attachment': 'free',
    'gill-spacing': 'close',
    'gill-size': 'broad',
    'gill-color': 'pink',
    'stalk-shape': 'equal',
    'stalk-root': 'bulbous',
    'stalk-surface-above-ring': 'smooth',
    'stalk-surface-below-ring': 'smooth',
    'stalk-color-above-ring': 'white',
    'stalk-color-below-ring': 'white',
    'veil-type': 'partial',
    'veil-color': 'white',
    'ring-number': 'one',
    'ring-type': 'pendant',
    'spore-print-color': 'brown',
    'population': 'numerous',
    'habitat': 'urban'
}
#kod do sprawdzenia, czy pieczara i kania dobrze pokaże jako jadalne
"""
mm = pd.DataFrame(test)
print(mm.head())


encoded_test = pd.get_dummies(mm)
cols = mushrooms_data_encoded.columns


for col in cols[1:]:
    if col not in encoded_test.columns:
        encoded_test[col] = 0

encoded_test = encoded_test[cols[1:]]


test_prediction = model_mushrooms.predict(encoded_test)
print(test_prediction)
"""