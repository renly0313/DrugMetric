# DrugMetric: Quantitative Drug-likeness Scoring Based on Chemical Space Distance


## Introduction

DrugMetric is a quantitative drug-likeness assessment model based on the distance within chemical space, designed to evaluate and screen compounds with potential medicinal value. By integrating chemical space distance, automated machine learning, and ensemble learning methods, DrugMetric demonstrates significant advantages in assessing drug-likeness, as well as robustness and stability in practical applications.

## Project Structure

This project comprises the following three directories:

1. **DrugMetric train and test**: Contains data and code for training and testing the DrugMetric model.
2. **High throughput virtual screening for 7kjs targets**: Contains experimental data and code for high-throughput virtual screening focused on the 7kjs target.
3. **Build DrugMetric web server**: Contains code and related files for constructing the DrugMetric web server.

## Key Features

- Quantitative evaluation of drug-likeness, allowing for convenient comparisons between the drug-likeness of different molecules.
- Utilization of Variational Autoencoders (VAEs) to learn the distribution characteristics of various datasets, and a mixed Gaussian model to differentiate between datasets with significant drug-likeness disparities.
- Calculation of chemical space distances to assign drug-likeness scores to datasets that initially lack drug-likeness labels.
- Incorporation of automated machine learning and ensemble learning concepts to attain an optimal drug-likeness scoring model.
- Demonstrated superior performance across multiple tasks and datasets, offering valuable insights for drug development.
- Development of an easy-to-use web server to facilitate use by medicinal chemists without a computational background.

## Results and Applications

- Achieved the highest AUROC scores (0.83, 0.94, 0.99) in three distinct tasks for distinguishing drugs from non-drugs.
- Accurately assessed the drug-likeness of nine molecular property prediction datasets in external datasets.
- Established significant correlations between DrugMetric scores and various drug attributes such as ADME properties, pharmacokinetics, toxicity, and molecular descriptors.
- Identified 10 potential CDK2 kinase inhibitors in virtual screening targeting CDK2 by integrating DrugMetric with molecular docking and other open-source tools for predicting CDK2 kinase inhibitors.

## Installation Dependencies

Ensure the following packages are installed:

- RDKit (version >= 2019)
- Python (version >= 3.8)
- PyTorch (version >= 1.8)
- autogluon (version >= 0.6.2)
- streamlit (version >= 1.14.0)

## Usage Instructions

### DrugMetric train and test

- Install the required packages.
- Download the pre-trained model.
- Use DrugMetric to evaluate the drug-likeness of input molecules.

### High throughput virtual screening for 7kjs targets

- Prepare the target structure and compound library.
- Execute the virtual screening workflow including molecular docking, scoring, and selection.
- Analyze the screening results to identify promising compounds.

### Build DrugMetric web server

- Install the necessary web server dependencies.
- Configure the server settings.
- Deploy and initiate the web server.
