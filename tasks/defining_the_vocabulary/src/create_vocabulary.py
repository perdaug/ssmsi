"""
Creating the vocabulary.

Terms:
    mni -- mass and intensity.

"""

import pymzml
import numpy as np
np.set_printoptions(precision=3, threshold=np.inf)

INTENSITY_FLOOR = 10
CLASS_TOLERANCE = 0.01
DATA_PATH = "../data/"
DATA_FILENAME = "abcdefgh_1.mzML"

def classify(classes, feature):
    for class_ in classes:
        if feature >= classes[class_][0] and feature <= classes[class_][1]:
            return class_
    print(feature)

def main():

    run = pymzml.run.Reader(DATA_PATH + DATA_FILENAME)

    # Iterating through the data and storing into a temporary array (`Pymzml Reader' has no length attribute)
    preprocessed_mnis = []  
    for spectrum in run:
        for mass, intensity in spectrum.peaks:
            if intensity > INTENSITY_FLOOR:
                preprocessed_mnis.append((mass, intensity, spectrum['id']))
    mnis_dtype = [('mass', float), ('intensity', float), ('id', int)]
    
    # Using numpy to sort the data.
    mnis = np.array(preprocessed_mnis, dtype=mnis_dtype)
    mnis.sort(order='mass') 
    
    # Defining mass ranges for words.
    words = {}
    starting_class_mass = mnis[0][0]
    previous_mass = mnis[0][0]
    for entity in mnis:
        mass = entity[0]
        if mass - previous_mass > CLASS_TOLERANCE:
            word = repr(starting_class_mass) + '--' + repr(previous_mass)
            words[word] = (starting_class_mass, previous_mass)
            starting_class_mass = mass
        previous_mass = mass 
    word = repr(starting_class_mass) + '--' + repr(previous_mass)
    words[word] = (starting_class_mass, previous_mass)

    # Populating the vocabulary
    vocabulary = {}
    for entity in mnis:
        key = entity[2]
        if key not in vocabulary:
            vocabulary[key] = {}
        class_ = classify(words, entity[0])
        if class_ not in vocabulary[key]:
            vocabulary[key][class_] = 0
        vocabulary[key][class_] += entity[1]

if __name__ == '__main__':
    main()
