from src.data import load_binarized_mnist

if __name__ == "__main__":
    print("Downloading and caching Binarized MNIST data...")
    # This function uses PennyLane's qml.data to download and cache the dataset
    load_binarized_mnist()
    print("Download complete.")
