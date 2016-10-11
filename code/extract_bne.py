"""
Extracting the letters b and e from the data sample.
"""

import math
import pymzml
import numpy as np
import matplotlib.pyplot as plt


ROW_LENGTH = 62
COLUMN_LENGTH = 1.25
COLUMN_HEIGHT = 10
NUMBER_OF_ROWS = 8

DATA_PATH = '../data/'

def main():

    run = pymzml.run.Reader(DATA_PATH + "abcdefgh_1.mzML")

    # Identifying the pixel idensity by extracting molecules required.
    pixel_intensities = []
    for spectrum in run:
        total_intensity = 0
        for mass, intensity in spectrum.peaks:
            if 375 <= mass <= 376:
                total_intensity += intensity
        pixel_intensities.append(total_intensity)


    # Calculating the row length.
    sample_size = len(pixel_intensities)
    total_row_length = ROW_LENGTH + COLUMN_LENGTH

    row_plus_column_size = math.ceil((sample_size / NUMBER_OF_ROWS))
    row_size = math.ceil(
        (sample_size / NUMBER_OF_ROWS) * (ROW_LENGTH / total_row_length))
   
    # Formatting the data for the picture.
    image_data = np.zeros(
        (NUMBER_OF_ROWS, row_size))

    column_idx = 0
    for idx, pixel_intensity in enumerate(pixel_intensities):
        row_idx = idx % row_plus_column_size
        # Cutting the column data.
        if row_idx < row_size:
            image_data[column_idx][row_idx] = pixel_intensity
        if row_idx == row_plus_column_size - 1:
            column_idx += 1

    # Reversing the rows required.
    for row_idx, row in enumerate(image_data):
        row_number = row_idx % NUMBER_OF_ROWS
        if row_number % 2 == 1:
            reversed_row = row[::-1]
            image_data[row_idx] = reversed_row


    plt.figure()
    plt.imshow(image_data, extent=[0, ROW_LENGTH, COLUMN_HEIGHT, 0])
    plt.show()

if __name__ == '__main__':
    main()
