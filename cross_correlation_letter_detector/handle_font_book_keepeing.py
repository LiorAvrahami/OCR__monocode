import os
from typing import Iterable
import re

from prompt_toolkit.completion import Completer, CompleteEvent, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.validation import Validator,ValidationError

def file_name_to_font_settings(db_file_path):
    name = os.path.splitext(os.path.basename(db_file_path))[0]
    name = name.split("dpi")[0]
    return tuple(name.split("_"))

def get_list_of_all_fonts_in_inventory(font_name_filter=None,font_size_filter=None,scanning_resolution_filter=None):
    fonts_dir_name = os.path.join(os.path.dirname(__file__), "all_calibrated_fonts")

    for db_file in os.listdir(fonts_dir_name):
        font_name,font_size,scanning_resolution = file_name_to_font_settings(db_file)

        # check filters
        if font_name_filter is not None and str.lower(font_name) != str.lower(font_name_filter):
            continue
        if font_size_filter is not None and font_size != font_size_filter:
            continue
        if scanning_resolution_filter is not None and scanning_resolution != scanning_resolution_filter:
            continue

        yield (font_name,font_size,scanning_resolution)

def inquire_font(font_name="",font_size="",scanning_resolution=""):
    try:
        from safer_prompt_toolkit import prompt
    except:
        from prompt_toolkit import prompt

    if font_name == "" and font_size == "" and scanning_resolution == "":
        font_name = "consolas"
        font_size = "8"
        scanning_resolution = "600"

    answer = prompt("select font settings, separated with comma [font_name,font_size,scanning_resolution]:\n",completer=FontCompleter(),validator=FontValidator(),
                    default=f"{font_name}, {font_size}, {scanning_resolution}")
    font_name,font_size,scanning_resolution = compile_user_font_selection(answer)

    return (font_name,font_size,scanning_resolution)

def compile_user_font_selection(answer,b_return_spans = False):
    match_res = re.match(" *(.*?) *, *(.*?) *, *(.*) *",answer)
    if match_res is None:
        match_res = re.match(" *(.*?) *, *(.*) *", answer)
    if match_res is None:
        match_res = re.match(" *(.*) *", answer)
    ret = match_res.groups()
    if not b_return_spans:
        # noinspection PyTypeChecker
        return list(ret) + [None]*(3-len(ret))
    else:
        # noinspection PyTypeChecker
        # return ((font_name, font_size, scanning_resolution),span_f,num_of_groups)
        return (list(ret) + [None] * (3 - len(ret)),match_res.span,match_res.lastindex)

class FontCompleter(Completer):

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        users_font_setting,span_f,num_of_groups = compile_user_font_selection(document.text,True)

        selected_setting_index = None
        selected_setting_span = None
        for i in range(1,num_of_groups + 1):
            if span_f(i)[0] <= document.cursor_position and span_f(i)[1] >= document.cursor_position:
                selected_setting_index = i - 1
                selected_setting_span = span_f(i)

        yield Completion(str(selected_setting_index))
        yield Completion(str(selected_setting_span))

        if selected_setting_index is not None and selected_setting_span is not None:
            users_font_setting[selected_setting_index] = None
            for possible_font_setting in get_list_of_all_fonts_in_inventory(*users_font_setting):
                yield Completion(possible_font_setting[selected_setting_index],
                                 selected_setting_span[0] - document.cursor_position) #todo make replace word when cursure is in middle of word. maybe lead - "fuzzycompleters"

class FontValidator(Validator):
    def validate(self, document: Document) -> None:
        (font_name, font_size, scanning_resolution) = compile_user_font_selection(document.text,False)
        error_text = "please enter font_name, font_size, scanning_resolution, separated by comma.\nfor example:\nconsolas, 8, 600"
        if font_name is None:
            raise ValidationError(cursor_position=0,message=error_text)
        if font_size is None:
            raise ValidationError(cursor_position=len(document.text),message=error_text)
        if scanning_resolution is None:
            raise ValidationError(cursor_position=len(document.text),message=error_text)

        if len(list(get_list_of_all_fonts_in_inventory(font_name, font_size, scanning_resolution))) == 0:
            raise ValidationError(message="no database file matches the selected font settings.\ndatabase files are found in cross_correlation_letter_detector/all_calibrated_fonts")



