import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from data_generator import load_captions, load_image, get_split_generators
from constants import seqs_to_captions


def main(count):

    accuracy_by_file = []
    for i in range(1, 6):
        filename = ".../data/captions_" + str(i) + ".txt"
        print("Current file: " + filename)

        data = load_captions(filename)
        data_dict = get_split_generators(data, batch_size=count)

        train_function = data_dict["train"]
        train_data_dict = train_function()
        X_train = train_data_dict[0]
        y_train = train_data_dict[1]

        test_function = data_dict["test"]
        test_data_dict = test_function()
        X_test = test_data_dict[0]
        y_test = test_data_dict[1]

        print("Images Dimension: " + str(X_train[0].shape) + ", Captions Dimension:" + str(y_train[0].shape))
        

        print("LinearRegression model created.")
        predictor = LinearRegression()
        print("LinearRegression model is fitting.")
        predictor.fit(X=X_train, y=y_train)
        print("LinearRegression model is creating output.")
        outcome = predictor.predict(X=X_test)
        print("LinearRegression model has created output.")

        print("Cleaning the output.")
        fixed_outcome = clean_output(outcome)
        
        print("Translating Sequences to Captions")
        written_caps = seqs_to_captions(y_test)
        written_preds = seqs_to_captions(fixed_outcome)
        correct, preds = filter(written_caps, written_preds)
        

        print("Determining accuracy")
        acc = generate_accuracy(correct, preds)
        accuracy_by_file.append(acc)

        print("\n")

    plot_acc(accuracy_by_file)
    overall_acc = np.mean(accuracy_by_file)
    print(overall_acc)
    
    
def plot_acc(accuracy_by_file):
    plt.style.use("ggplot")
    plt.figure()
    
    plt.plot(["captions_1", "captions_2", "captions_3", "captions_4", "captions_5"], accuracy_by_file)

    plt.ylim(0, .05)
    plt.title("Training Accuracy")
    plt.xlabel("Captions File Number")
    plt.ylabel("Accuracy")
    plt.savefig(".../data/baseline_acc.png")

def clean_output(outcome):
    fixed_outcome = np.zeros((outcome.shape[0], outcome.shape[1]))
    for row in range(outcome.shape[0]):
        for col in range(outcome.shape[1]):
            num = np.absolute(np.round(outcome[row][col]))
            if num >= 1623:
                num = 0
            fixed_outcome[row][col] = num
    return fixed_outcome

def flatten_images(images):
    image_dim = images.shape
    X = np.zeros((image_dim[0], image_dim[1]*image_dim[2]*image_dim[3]))
    for i in range(image_dim[0]):
        flattened_image = images[i].flatten()
        X[i] = flattened_image
    
    return X

def output_csv(written_caps, written_preds):
    captions_df = pd.DataFrame(columns=["Correct_Caption", "Predicted_Caption"])
    for i in range(len(written_caps)):
        captions_df = captions_df.append({'Correct Caption': written_caps[i], 'Predicted_Caption': written_preds[i]}, ignore_index=True)

    captions_df.to_csv('...data/CaptionsVsPredictions.csv')


def filter(correct, predictions):
    filtered_correct = []
    filtered_preds = []

    i = 0
    while i < len(correct):
        filtered_correct.append(correct[i])
        filtered_preds.append(predictions[i])
        i += len(correct[i]) + 1
    
    return filtered_correct, filtered_preds


def generate_accuracy(correct, predictions):
    accuracies = []
    for i in range(len(correct)):
        correct_words = correct[i].split()
        predicted_words = predictions[i].split()

        overlap = set(correct_words) & set(predicted_words)

        if(len(correct_words) > 0):
            acc = len(overlap)/len(correct_words)
            accuracies.append(acc)
    
    return np.mean(accuracies)

if __name__ == "__main__":
    main(count=2000)

