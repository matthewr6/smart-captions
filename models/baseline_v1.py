import numpy as np
import pandas as pd
import collections

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from data_generator import data_generator
from constants import seqs_to_captions


def main(count):
    gen = data_generator(batch_size=count, mode='not vgg16')
    images, captions, next_words = next(gen)
    print("Images Dimension: " + str(images.shape) + ", Captions Dimension:" + str(captions.shape) + ", Next Words Dimension: " + str(next_words.shape))
    
    X = flatten_images(images)
    y = captions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

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
    
    print("Outputing the Captions side-by-side to a dataframe")
    output_csv(written_caps, written_preds)

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


if __name__ == "__main__":
    main(count=200)

