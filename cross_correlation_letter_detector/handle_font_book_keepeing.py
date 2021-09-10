
def get_list_of_all_fonts_in_inventory(font_name=None,font_size=None,scanning_resolution=None):
    return ["Consolas8_600dpi"] #Todo implement

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



