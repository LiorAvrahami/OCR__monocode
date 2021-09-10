import matplotlib.pyplot as plt, numpy as np
import typing
from cross_correlation_letter_detector.do_cross_correlation_v1 import do_ccr_v1
import pickle
import os
from handle_font_book_keepeing import get_list_of_all_fonts_in_inventory

def do_ccr(path,font=None,frame_character = "X",corner_charicter="O") -> (float,str):
    if font is None:
        max_score_font = (float("-inf"),None,None)
        for font_size in get_list_of_all_fonts_in_inventory():
            total_score,text = do_ccr(path,font_size)
            if total_score > max_score_font[0]:
                max_score_font = (total_score,text,font_size)
        print(f"selected font size: {max_score_font[2]}")
        return max_score_font[0],max_score_font[1]
    letters_db = load_character_map(font)
    path_for_dump_files = os.path.splitext(path)[0] + "_ocr_crosscorrelation_results"
    return do_ccr_v1(path,letters_db,path_for_dump_files,frame_character,corner_charicter)


def load_character_map(font) -> typing.List[typing.Tuple[str,np.ndarray]]:
    """
    :param font_size: a string, consisting of f"{font_type}{font_size}_{resolution_dpi}dpi" for example "Consolas8_600dpi"
    :return: a sorted list of tuples containing a character and an array of images of that character. sorted by character order alphabetically
    """
    dir_name = os.path.dirname(__file__)
    data_bank_file_name = os.path.join(dir_name,f"{font}_db.pickle")
    with open(data_bank_file_name, "rb") as f:
        letters_db = pickle.load(f)
    return letters_db