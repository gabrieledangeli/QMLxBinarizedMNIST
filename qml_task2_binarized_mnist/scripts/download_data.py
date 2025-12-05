from src.data import load_binarized_mnist

if __name__ == "__main__":
    print("Downloading and caching Binarized MNIST data...")
    # This function uses sklearn's fetch_openml which caches data by default in ~/scikit_learn_data
    load_binarized_mnist()
    print("Download complete.")
