import pandas as pd
from ydata_profiling import ProfileReport
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from config import vars

# https://github.com/Mati0106/ML_python/tree/main/lecture_one
# https://www.kaggle.com/datasets/volodymyrgavrysh/fraud-detection-bank-dataset-20k-records-binary/code

df = pd.read_csv(
    vars['file_name'],
    index_col=0)  # Column 0 has only indexes

# check basic parameters for dataset
print(df.shape)
print(df.head)
print(df.describe())
# divide dataset for frauds and no_frauds
no_fraud = df[(df["targets"] == 0)]
fraud = df[df["targets"] == 1]
print("Number of frauds in database: " + str(fraud.shape[0]))
print("% of frauds in database: " + str("{:.1%}".format(fraud.shape[0] / df.shape[0])))

# check missing values
missing_count = df.isnull().any().sum() + df.isna().sum().sum()
print(f'Count of features with missing values: {missing_count}')

# delete columns with variance == 0 (keeping columns with variance>0)
df = df.loc[:, (df.var() > 0)]
print(df.shape)


# https://www.kaggle.com/code/oldwine357/removing-highly-correlated-features
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x


df = remove_collinear_features(df, vars['threshold_corr'])


report = ProfileReport(df, title='My Data', correlations={
    "pearson": {"calculate": True},
    "spearman": {"calculate": True},
    "kendall": {"calculate": True}
  })
report.to_file("ProfileReport.html")


# drop very imbalanced columns based on ProfileReport: col_28 col_58 col_108
df = df.drop(columns=['col_28', 'col_58', 'col_108'])

# Standardization
# separate the independent and dependent variables
scale = StandardScaler()
Y = df["targets"]
X = df.drop("targets", axis=1)

# standardization of dependent variables
df = scale.fit_transform(X)
print(df)

# PCA
pca = PCA(n_components=2)
pca.fit(X)

pca.transform(X)

pca = PCA()
components = pca.fit_transform(X)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=Y
)
fig.update_traces(diagonal_visible=False)
fig.show()

# modeling

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12)

n_estimators = 10
max_depth = 3

Logistic_Regression = LogisticRegression(solver="liblinear", class_weight={1: 2}, random_state=10)
Logistic_Regression.fit(X_train, y_train)

random_forest = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10)
random_forest.fit(X_train, y_train)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=10)
_ = gradient_boosting.fit(X_train, y_train)

models = [
    ("RF", random_forest),
    ("LR", Logistic_Regression),
    ("GBDT", gradient_boosting),
]

for name, model in models:
    precision = precision_score(y_test, model.predict(X_test))
    print("precision of {}:{}".format(name, precision))
