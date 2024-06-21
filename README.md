# Project_4

In this project I delve into the practical application of machine learning on the field of biomedicine, specifically in it;s relation to novel drug development fir the treatment of leukemia (i.e. chronic myelogenous leukemia).

## Background
Cancer remains a major global health challenge, affecting millions of people worldwide. Despite significant advancements in cancer research and treatment, many forms of cancer still have limited treatment options and can be difficult to cure. Conventional cancer therapies, such as chemotherapy and radiation, often come with severe side effects that can greatly impact a patient's quality of life. As a result, there is a pressing need for the development of new, more effective, and less toxic anti-cancer drugs.

One promising avenue for discovering novel anti-cancer compounds is through the exploration of plant-based sources. Plants have been used for medicinal purposes for centuries, and many modern pharmaceuticals have been derived from plant compounds. Plants produce a wide variety of secondary metabolites, which are chemical compounds that are not essential for the plant's growth and development but may play important roles in defense against herbivores, pathogens, or environmental stresses. These secondary metabolites have been found to possess various biological activities, including anti-inflammatory, antioxidant, and anti-cancer properties.

Research has shown that many plant-derived compounds have the potential to selectively target cancer cells while minimizing harm to healthy cells, making them promising candidates for the development of new anti-cancer drugs. Furthermore, plants are a renewable and economically viable source of these compounds, making them an attractive option for drug discovery.

As scientists continue to investigate the anti-cancer potential of plant-based compounds, there is hope that new, more effective, and less toxic treatments can be developed to improve the lives of cancer patients worldwide. By harnessing the power of nature and combining it with modern scientific techniques, we may be able to unlock the full potential of plant-based medicine in the fight against cancer.

## Objective

Drug development is naturally a long and arduous process, with the discovery of a lead compound with the desired effect being the crucial first step. Our machine learning (ML) model aims to assist biomedical researchers in this early stage by predicting the IC50 value of novel compounds specifically targeting the BCR-ABL receptor in Chronic Myelogenous Leukemia (CML). By providing these predicted values, our model can help guide researchers in determining the actual IC50 value through in-vitro experiments, potentially streamlining the initial screening process. However, it is important to note that our model's predictions are limited to the BCR-ABL receptor and may not be applicable to other receptors. The IC50 values, being numerical in nature, can be effectively predicted using regression models, which form the foundation of our solution.

## Data Collection and Preparation

The dataset for this project was obtained from the [ChemBL database of the European Bioinformatic Institute](https://www.ebi.ac.uk/chembl/). We searched for BCR-ABL as the target, filtered for the human BCR-ABL gene, and selected only compounds with IC50 values towards BCR-ABL. After downloading, the dataset underwent a comprehensive chemoinformatics pipeline for preparation. This included generating 'canonical' SMILES from 'SMILES' and calculating 200 molecular descriptors using the RDKit library. Further preprocessing steps involved scaling, train-test splitting, and clustering. In the post-processing phase, we employed Recursive Feature Elimination with Cross-Validation (RFE-CV) and Principal Component Analysis (PCA) to refine our feature set and reduce dimensionality, respectively.

### RDKit

RDKit ([Github](https://github.com/rdkit/rdkit)/[Offcial site](https://www.rdkit.org])) is an open source python library designed to do chemoinformatics and machine learning related pipeline in medicinal chemistry. In this context, the most important use of RDKit is to generate molecular descriptors (mathematical representation of molecular features) from their respective SMILES. Before calculating the descriptor several step must be undertaken to ensure a proper data for training\

### Canonical SMILES generation

SMILES should be discrete for every molecule. Sometimes the SMILES are written slightly different but refers to the same molecule. Canonical SMILES generation eliminate the possibility of the same molecule having slightly different SMILES. Imagine, someone write 'sugar' as 'sugAr', both refer to the same thing, which is sugar, but the algorithm identifies sugar and sugAr as different molecules. Canonical SMILES generation eliminates 'sugAr' and assigned it also as 'sugar'

this step also use RDKit especially the ```def canonical_smiles(smiles):``` as outlined below

```
def canonical_smiles(smiles):
   mols = [Chem.MolFromSmiles(smi) for smi in smiles]
   canonical_smiles = [Chem.MolToSmiles(mol) for mol in mols]
   return canonical_smiles

df['canonical'] = canonical_smiles(df['smiles'])
```

### Duplicate handling

Molecular duplicates are listed and dropped to prevent redundancy 

```
duplicate = df[df['canonical'].duplicated()]['canonical'].values
len(duplicate)
```
```
df[df['canonical'].isin(duplicate)].sort_values(by='canonical')
```
```
df = df.drop_duplicates(subset='canonical')
```

### Descriptor calculation

By design, RDKit can calculate about 200 descriptors ([documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors)) using this function

```
def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names
```
```
#function calling

Mol_descriptors,desc_names = RDkit_descriptors(df['canonical'])
```

After this step, the dataset can be processed through regular machine learning pipeline

### Train-test split

Train-test is done in a straighforward manner. Please refer to the code snippet for clarity

```
X = df.drop(columns=['ic50', 'id', 'smiles', 'unit', 'canonical'])
y = df['ic50']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

### Scaling

Scaling is done to all 200 descriptors using standard scaler. This step can be experimented upon for the best result

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) # fit and transform the training dataset
X_test_scaled = scaler.transform(X_test) # only transform the test set

```
This is an especially important step particularly when using SVM or NN algorithms as it will be explaineed below

### Clustering

Due to the high-dimensionality of data, there is a need for dimensionality reduction. We employ DBSCAN to cluster the features. Before actually applyind DBSCAN it's proper to determine the 'epsilon' value. To do this we use k-distance graph

```
# k-distance graph to find a proper 'eps' for DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

k = 4

neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(X_train_scaled)
distances, indices = nbrs.kneighbors(X_train_scaled)

k_distances = np.sort(distances[:, k-1])

plt.plot(range(1, len(k_distances) + 1), k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-distance')
plt.title(f'k-distance Graph for k={k}')
plt.show()
```
![k-distance graph](https://example.com/path/to/image.png)

As shown the elbow point is at y=10, thus epsilon is determined to be 10

```
dbscan = DBSCAN(eps=10, min_samples=5)
clusters = dbscan.fit_predict(X_train_scaled)

unique_labels = set(clusters)
n_clusters = len(unique_labels) - (1 if -1 in clusters else 0)
n_noise_points = list(clusters).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise_points}")
```
note: please refer to the attached Jupyter Notebook for the cluster visualization

From this result we can separate the 'noise' from the non noise

```
from sklearn.model_selection import train_test_split

is_noise = (clusters == -1)
X_noise = X_train_scaled[is_noise]

```
note: Due to time constraint, 'X_noise' was left alone. Theoretically this could be used to derive new features, and can be used as a training dataset for model specific to the noise.

## Base Model Training

The first round of training consists of training five different models and determine which from those 5 is best. Granted, the choice of what model to use here is somewhat arbitrary, empirically at best, thus 'unscientific;. Here I will outline the reason for choosing the models

* Linear regression: as we're trying to solve a regression problem, the simplest model which is linear regression is chosen and could act as a baseline model to compare the other models to
* Random Forest (RF): clasically a very effective model for regression and offers robustness and good generalization. They handle nonlinear relationships quite well and provide measures of feature importance, which can be very enlightening
* Gradient Boosting Machines (GBM) i.e. XGBoost: work well for datasets with mixed data types and are generally considered to be more efficient in terms of computational resources
* Support Vector Machines w/ Regression (SVM-R): able to deal with complex relationships that might not be linear, it's highly probable that this is the case considering the tremendous amount of features
* Neural Networks i.e. Multi-Layer Perceptron (MLP): 'cutting edge' and proven in various bioinformatics application albeit not necessarily related to this case

From this step SVM-R and MLP are apparently better than the rest, using MSE and R2 score as the evaluation metric
note: for detailed results please refer to the attached Jupyter Notebook

## Post-processing

This step carries over the previous best two models to be processed using recursive feature elimination (RFE). Once again the 'estimator' and 'n_features_to_select' was decided arbitrarily

```
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

estimator = LinearRegression() # can be experimented upon

rfe = RFE(estimator=estimator, n_features_to_select=50)  # 'n' can be experimented upon

rfe.fit(X_train, y_train)
```

the RFE-fitted dataset was then fitted to PCA with PCA(n-components=0.95). 

```
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # can be changed

X_train_pca = pca.fit_transform(X_train_scaled)
print(f'Number of components selected: {pca.n_components_}') # inspect number of components

```
```
X_test_pca = pca.transform(X_test_scaled)
```
note: the decision to do PCA right after RFE was done yet again arbitrarily and due to time constraint, and might not be commonly done in a proper ML pipeline 

## Final Model Training

Using the dataset that has been reduced by PCA

for SVR

```
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

svm_pipeline = Pipeline([
    ('svm', SVR())
])

svm_pipeline.fit(X_train_pca, y_train)

y_pred_svm = svm_pipeline.predict(X_test_pca)
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print(f"SVM Mean Squared Error: {mse_svm:.2f}")
print(f"SVM R^2 Score: {r2_svm:.2f}")
```
and for MLP
```
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
])

# Train
mlp_pipeline.fit(X_train_pca, y_train)

y_pred_mlp = mlp_pipeline.predict(X_test_pca)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(f"MLP Mean Squared Error: {mse_mlp:.2f}")
print(f"MLP R^2 Score: {r2_mlp:.2f}")
```

Note: please refer to the attached Jupyter Notebook for the results


