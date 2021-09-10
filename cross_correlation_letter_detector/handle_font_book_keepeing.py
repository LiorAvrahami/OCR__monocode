import os

def file_name_to_font_settings(db_file_path):
    name = os.path.splitext(os.path.basename(db_file_path))[0]
    name = name.split("dpi")[0]
    return name.split("_")

def get_list_of_all_fonts_in_inventory(font_name_filter=None,font_size_filter=None,scanning_resolution_filter=None):
    fonts_dir_name = os.path.join(os.path.dirname(__file__), "all_calibrated_fonts")

    for db_file in fonts_dir_name:
        font_name,font_size,scanning_resolution = file_name_to_font_settings(db_file)

        # check filters
        if font_name_filter is not None and str.lower(font_name) == str.lower(font_name_filter):
            continue
        if font_size_filter is not None and font_size == font_size_filter:
            continue
        if scanning_resolution_filter is not None and scanning_resolution == scanning_resolution_filter:
            continue

        yield (font_name,font_size,scanning_resolution)

def inquire_font(font_name=None,font_size=None,scanning_resolution=None):
    try:
        from safer_prompt_toolkit import prompt
    except:
        from prompt_toolkit import prompt

    # todo completers

    if font_name is None:
        font_name = prompt("select font name")
    if font_size is None:
        font_size = int(prompt("select font size"))
    if scanning_resolution is None:
        scanning_resolution = int(prompt("select scanning resolution"))

    return (font_name,font_size,scanning_resolution)



