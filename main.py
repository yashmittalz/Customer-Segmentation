import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from kmodes.kprototypes import KPrototypes

# Function to load and preprocess the dataset
def load_and_preprocess_data():
    # Load dataset
    try:
        df = pd.read_csv("/Users/yashmittal/Documents/Github/Customer Segmentation/segmentation-data.csv")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: Dataset file not found. Make sure 'segmentation-data.csv' is in the same directory.")
        return None

    # Check for null values
    if df.isnull().sum().any():
        print("Error: Dataset contains null values. Please clean the dataset before proceeding.")
        return None

    # Normalize numerical columns
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Income'] = scaler.fit_transform(df[['Income']])

    # Drop unnecessary columns
    df_temp = df[['ID', 'Age', 'Income']]
    df = df.drop(['ID'], axis=1)

    # Convert data types for KPrototypes
    mark_array = df.values
    mark_array[:, 2] = mark_array[:, 2].astype(float)
    mark_array[:, 4] = mark_array[:, 4].astype(float)

    return df, df_temp, mark_array

# Function to perform clustering
def perform_clustering(mark_array, n_clusters):
    kproto = KPrototypes(n_clusters=n_clusters, verbose=2, max_iter=20)
    clusters = kproto.fit_predict(mark_array, categorical=[0, 1, 3, 5, 6])
    return kproto, clusters

# Function to visualize clusters
def visualize_clusters(df, n_clusters):
    colors = ['green', 'red', 'gray', 'orange', 'yellow', 'cyan', 'magenta', 'brown', 'purple', 'blue']
    plt.figure(figsize=(15, 15))
    plt.xlabel('Age')
    plt.ylabel('Income')

    for i, col in zip(range(n_clusters), colors):
        dftemp = df[df.cluster == i]
        plt.scatter(dftemp.Age, dftemp['Income'], color=col, alpha=0.5, label=f"Cluster {i}")

    plt.legend()
    plt.show()

# Menu-driven application
def main():
    while True:
        print("\nCustomer Segmentation App")
        print("1. Load and preprocess data")
        print("2. Perform clustering")
        print("3. Visualize clusters")
        print("4. Exit")

        try:
            choice = input("Enter your choice: ")
        except OSError:
            print("Error: Input is not supported in this environment. Please provide inputs programmatically.")
            return

        if choice == '1':
            global df, df_temp, mark_array
            data = load_and_preprocess_data()
            if data:
                df, df_temp, mark_array = data
                print("Data loaded and preprocessed successfully.")
        elif choice == '2':
            if 'mark_array' not in globals():
                print("Error: Load and preprocess data first.")
                continue
            try:
                n_clusters = int(input("Enter the number of clusters: "))
            except OSError:
                print("Error: Input is not supported in this environment. Please set 'n_clusters' programmatically.")
                return
            global kproto, clusters
            kproto, clusters = perform_clustering(mark_array, n_clusters)
            df['cluster'] = clusters
            df[['ID', 'Age', 'Income']] = df_temp
            print(f"Clustering completed with {n_clusters} clusters.")
        elif choice == '3':
            if 'df' not in globals() or 'clusters' not in globals():
                print("Error: Perform clustering first.")
                continue
            n_clusters = len(set(clusters))
            visualize_clusters(df, n_clusters)
        elif choice == '4':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
