#构建测试集
import argparse
import pickle
import numpy as np
import pandas as pd

def generate_test_set(drug_path, chembl_path, zinc_path, gdb_path):
    # Load the mean vectors
    with open(drug_path, "rb") as f:
        drug_features = pickle.load(f, encoding='latin1')
    with open(chembl_path, "rb") as f:
        chembl_features = pickle.load(f, encoding='latin1')
    with open(zinc_path, "rb") as f:
        zinc_features = pickle.load(f, encoding='latin1')
    with open(gdb_path, "rb") as f:
        gdb_features = pickle.load(f, encoding='latin1')

    # Generate class scores for each row
    drug_scores = np.clip(np.random.normal(100, 19.78, size=drug_features.shape[0]), 0, 100)
    chembl_scores = np.clip(np.random.normal(28.62, 10.53, size=chembl_features.shape[0]), 0, 100)
    zinc_scores = np.clip(np.random.normal(22.83, 15.37, size=zinc_features.shape[0]), 0, 100)
    gdb_scores = np.clip(np.random.normal(0, 15.61, size=gdb_features.shape[0]), 0, 100)

    # Stack the class scores to create a column vector
    Y = np.hstack((drug_scores, chembl_scores, zinc_scores, gdb_scores))

    # Stack the feature matrices horizontally and add the class scores as the last column
    X = np.vstack((drug_features, chembl_features, zinc_features, gdb_features))

    # Convert X to a pandas DataFrame
    X_df = pd.DataFrame(X)

    # # Convert Y to a pandas DataFrame and concatenate it with X_df
    Y_df = pd.DataFrame(Y)
    data = pd.concat([X_df, Y_df], axis=1)

    return data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', type=str, default='../data/latent_space_result/test_mean_1206_l64/seprated_clinical_drug_test_mean.pkl', help='Path to drug mean pickle file')
    parser.add_argument('--chembl', type=str, default='../data/latent_space_result/test_mean_1206_l64/seprated_chembl_test_rand1_mean.pkl', help='Path to chembl mean pickle file')
    parser.add_argument('--zinc', type=str, default='../data/latent_space_result/test_mean_1206_l64/seprated_zinc_test_rand1_mean.pkl', help='Path to zinc mean pickle file')
    parser.add_argument('--gdb', type=str, default='../data/latent_space_result/test_mean_1206_l64/seprated_gdb_test_rand1_mean.pkl', help='Path to gdb mean pickle file')
    parser.add_argument('--output', type=str, default='../autogluen/test1.csv', help='Path to output csv file')

    args = parser.parse_args()

    # Generate the training set
    test_set = generate_test_set(args.drug, args.chembl, args.zinc, args.gdb)

    # Save the training set to a CSV file
    test_set.to_csv(args.output, index=False)