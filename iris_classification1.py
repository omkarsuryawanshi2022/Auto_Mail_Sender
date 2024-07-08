from sklearn.datasets import load_iris

def main():
    print("Iris flower case study----------")

    data = load_iris()

    Feature = data.data
    Labels = data.target

    print("Features are:")
    print(Feature)

    print("Labels are:")
    print(Labels)

if __name__ == "__main__":
    main()
