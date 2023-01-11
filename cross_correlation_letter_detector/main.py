import numpy as np
import typing

from safer_prompt_toolkit import make_ConstantOptions_Completer_and_Validator, prompt

from cross_correlation_letter_detector.do_cross_correlation_v2 import do_ccr_v2
import pickle
import os
from .handle_font_book_keepeing import get_list_of_all_fonts_in_inventory, inquire_font


def do_ccr(path, font_settings=None, b_has_frame=True, frame_character="X", corner_charicter="O", b_load_last_run=True, b_skip_ocr=False) -> (float, str):
    # TODO use b_has_frame

    # ask user which font to use
    if font_settings is None:
        options = ["run all fonts", "select font"]
        completer, validator = make_ConstantOptions_Completer_and_Validator(options)
        answer = prompt("would you like to run all fonts and use the best matching one, or would you like to select specific font? ", completer=completer,
               validator=validator)
        assert answer in options
        if answer == options[0]:
            font_settings = None
        else:
            font_settings = inquire_font()
    if any([val == "" or val is None for val in font_settings]):
        font_settings = inquire_font(*font_settings)

    # run for all fonts and use best matching one
    if font_settings is None:
        max_score_font = (float("-inf"), None, None)
        for font_settings in get_list_of_all_fonts_in_inventory():
            total_score, text = do_ccr(path, font_settings)
            if total_score > max_score_font[0]:
                max_score_font = (total_score, text, font_settings)
        print(f"selected font settings: {max_score_font[2][0]}{max_score_font[2][1]} {max_score_font[2][2]}dpi")
        return max_score_font[0], max_score_font[1]

    letters_db = load_character_map(font_settings)
    path_for_dump_files = os.path.splitext(path)[0] + "_ocr_crosscorrelation_results"
    return do_ccr_v2(path, letters_db, path_for_dump_files, frame_character, corner_charicter, b_load_last_run, b_skip_ocr=b_skip_ocr)


def load_character_map(font_settings) -> typing.Dict[str, np.ndarray]:
    """
    :param font_size: a string, consisting of f"{font_type}{font_size}_{resolution_dpi}dpi" for example "Consolas8_600dpi"
    :return: a sorted list of tuples containing a character and an array of images of that character. sorted by character order alphabetically
    """
    fonts_dir_name = os.path.join(os.path.dirname(__file__), "all_calibrated_fonts")
    data_bank_file_name = os.path.join(fonts_dir_name, f"{str.lower(font_settings[0])}_{font_settings[1]}_{font_settings[2]}dpi_db.pickle")
    with open(data_bank_file_name, "rb") as f:
        letters_db = pickle.load(f)
    return letters_db
