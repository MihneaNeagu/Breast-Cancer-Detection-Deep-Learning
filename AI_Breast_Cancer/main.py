import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from sklearn.metrics import confusion_matrix


def load_data(file_path):
    dt = pd.read_csv(file_path)
    dt.drop(dt.columns[32], axis=1, inplace=True)
    print(dt)
    return dt


def plot_countplot(dt):
    custom_palette = {"M": "red", "B": "green"}
    sns.countplot(x="diagnosis", data=dt, palette=custom_palette)
    plt.show()


def encode_diagnosis(dt):
    y = dt['diagnosis'].values
    print("Diagnosis before encoding are: ", np.unique(y))
    labelencoder = LabelEncoder()
    Y = labelencoder.fit_transform(y)
    print("Labels after encoding are: ", np.unique(Y))
    return Y


def preprocess_data(dt):
    X = dt.drop(labels=["diagnosis", "id"], axis=1)
    print(X.describe().T)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print(X)
    return X


def split_data(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print("Training data is:", X_train.shape)
    print("Testing data is:", X_test.shape)
    return X_train, X_test, y_train, y_test

from keras.layers import LeakyReLU
from keras.activations import tanh
def build_model():
    # Defining the model
    model = Sequential()
    model.add(Dense(32, input_dim=30, activation='relu'))  # Input layer with 32 neurons
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons
    model.add(Dense(64, activation='relu'))  # Additional hidden layer with 64 neurons
    model.add(Dense(32, activation='relu'))  # Additional hidden layer with 32 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (binary classification)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, verbose=1, epochs=150, batch_size=32,
                        validation_data=(X_test, y_test))
    return history


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'm', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    confusion_matr = confusion_matrix(y_test, y_pred)
    color_palette = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(confusion_matr, annot=True, cmap=color_palette)
    plt.show()


def main():
    file_path = "data/breast-cancer-wisconsin-data_data.csv"

    while True:
        print("Menu:")
        print("1. Load Data")
        print("2. Plot Countplot")
        print("3. Encode Diagnosis")
        print("4. Preprocess Data")
        print("5. Split Data")
        print("6. Build Model")
        print("7. Train Model")
        print("8. Plot Loss")
        print("9. Plot Accuracy")
        print("10. Evaluate Model")
        print("11. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            dt = load_data(file_path)
        elif choice == "2":
            if 'dt' not in locals():
                print("Please load data first.")
                continue
            plot_countplot(dt)
        elif choice == "3":
            if 'dt' not in locals():
                print("Please load data first.")
                continue
            Y = encode_diagnosis(dt)
        elif choice == "4":
            if 'dt' not in locals():
                print("Please load data first.")
                continue
            X = preprocess_data(dt)
        elif choice == "5":
            if 'X' not in locals() or 'Y' not in locals():
                print("Please preprocess data first.")
                continue
            X_train, X_test, y_train, y_test = split_data(X, Y)
        elif choice == "6":
            if 'X_train' not in locals() or 'y_train' not in locals():
                print("Please split data first.")
                continue
            model = build_model()
        elif choice == "7":
            if 'model' not in locals() or 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
                print("Please build model and split data first.")
                continue
            history = train_model(model, X_train, y_train, X_test, y_test)
        elif choice == "8":
            if 'history' not in locals():
                print("Please train model first.")
                continue
            plot_loss(history)
        elif choice == "9":
            if 'history' not in locals():
                print("Please train model first.")
                continue
            plot_accuracy(history)
        elif choice == "10":
            if 'model' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
                print("Please build model and split data first.")
                continue
            evaluate_model(model, X_test, y_test)
        elif choice == "11":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number from the menu.")


if __name__ == "__main__":
    main()


