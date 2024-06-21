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

The dataset for this project was obtained from the ChemBL database of the European Bioinformatic Institute (https://www.ebi.ac.uk/chembl/). We searched for BCR-ABL as the target, filtered for the human BCR-ABL gene, and selected only compounds with IC50 values towards BCR-ABL. After downloading, the dataset underwent a comprehensive chemoinformatics pipeline for preparation. This included generating 'canonical' SMILES from 'SMILES' and calculating 200 molecular descriptors using the RDKit library. Further preprocessing steps involved scaling, train-test splitting, and clustering. In the post-processing phase, we employed Recursive Feature Elimination with Cross-Validation (RFE-CV) and Principal Component Analysis (PCA) to refine our feature set and reduce dimensionality, respectively.
