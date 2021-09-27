from scipy.signal import correlate
import numpy as np

def normalise_kernel_in_place(ker):
    if np.std(ker) > 10**-10:
        ker -= np.mean(ker)
    ker /= np.sum(ker ** 2) ** 0.5

def calculate_normalised_crosscorrelation(document_image,char_image,b_try_again_if_fail=True):
    document_image = document_image - np.mean(document_image)
    crcr = correlate(document_image, char_image, mode="same")
    norm = correlate(document_image**2, char_image*0 + 1, mode="same")**0.5
    if np.any(np.isnan(norm)):
        if b_try_again_if_fail:
            print(f"\nweird random math error!? at file {__file__}")
            print("trying calculation again..")
            return calculate_normalised_crosscorrelation(document_image,char_image,b_try_again_if_fail=False)
        else:
            raise ValueError("weird random math error!? at file {__file__}")
    fitv = crcr/norm
    return fitv