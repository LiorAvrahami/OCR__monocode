from scipy.signal import correlate
import numpy as np

def normalise_kernel_in_place(ker):
    if np.std(ker) > 10**-10:
        ker -= np.mean(ker)
    ker /= np.sum(ker ** 2) ** 0.5

def calculate_normalised_crosscorrelation(document_image,char_image):
    document_image = document_image - np.mean(document_image)
    crcr = correlate(document_image, char_image, mode="same")
    norm = correlate(document_image**2, char_image*0 + 1, mode="same")**0.5
    fitv = crcr/norm
    return fitv