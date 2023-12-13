from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from config import vars
# https://github.com/Mati0106/ML_python/tree/main/lecture_one
# https://www.kaggle.com/datasets/volodymyrgavrysh/fraud-detection-bank-dataset-20k-records-binary/code
#zmiana
p = Path('.')
[x for x in p.iterdir() if x.is_dir()]
df = pd.read_csv(
    "C:/Users/rober/Desktop/Studia PG/2 semestr/Uczenie maszynowe w pythonie Serocki/Projekt/fraud_detection_bank_dataset.csv",
    index_col=0)  # Column 0 has only indexes
#inaczej zrobić te odniesienie do danych

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

"""
report = ProfileReport(df, title='My Data',correlations = {
    "pearson": {"calculate": True},
    "spearman": {"calculate": True},
    "kendall": {"calculate": True}
  })
report.to_file("my_report.html")
"""
# drop very imbalanced columns based on ProfileReport: col_28 col_58 col_108
# co z innymi niezbalansowanymi i o wysokiej korelacji?
df = df.drop(columns=['col_28', 'col_58', 'col_108'])

# standardscaler, a nie minmax
# Standardization. Using MaxMinScaler as the data is skewed # https://www.askpython.com/python/examples/normalize-data-in-python
scaler = MinMaxScaler()
df_std = pd.DataFrame(scaler.fit_transform(df),
                      columns=df.columns, index=df.index)
"""
# show plots for columns in df_std
fig, axes = plt.subplots(37, 2, figsize=(30, 15))
# Flatten the axes array to easily iterate over the subplots
axes = axes.flatten()

for col, ax in enumerate(axes):
    sns.kdeplot(data=df_std, x=df.columns[col], fill=True, ax=ax, warn_singular=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(df.columns[col], loc='left', weight='bold', fontsize=10, pad=-10)
plt.show()
"""


# definiujemy ile komponentow chcemy otrzymac z PCA (ile wyznaczyc wektorow wlasnych, patrz liczenie PCA z zajec nr1, link na wykladzie)
pca = PCA(n_components=1)

X_train = pca.fit_transform(df)

explained_variance = pca.explained_variance_ratio_
print(pca)
print(X_train)
print(explained_variance)