# ## file documentation:
# this file contains the function that iterates over all the files in the given root directory, and copies their content one after an other (with a header
# containing the file name, font, and page dimensions). this function is called by the OCR_monocode.py main script.

import os, copy, efipy

__version__ = "1.0"


def make_file_header(name, font_name, font_size, page_width, page_height):
    return f"$${name}$$ Font:{font_name}{font_size} | PageWidth:{page_width} | PageHeight:{page_height}"


def put_page_in_frame(page,page_width,page_height,corner_char="O", frame_char="X"):
    top_frame = corner_char + frame_char * page_width + corner_char
    bottom_frame = top_frame
    right_frame_char = frame_char
    left_frame_char = frame_char
    page += [""] * (page_height - len(page))

    # right left bounds
    for i in range(len(page)):
        page[i] = left_frame_char + page[i] + " " * (page_width - len(page[i])) + right_frame_char

    # top bottom bounds
    if top_frame != "":
        page.insert(0, top_frame)
    if bottom_frame != "":
        page.append(bottom_frame)


def file_tree_to_text_file(input_path, output_path, b_has_frame):
    excluded_file_types = [".docx", ".wbk"]

    font_name = "ConSolas"
    font_size = 8
    page_width = 125
    page_height = 77

    total_text = ""

    def handle_file(path):
        global total_text
        if os.path.splitext(path)[1] in excluded_file_types:
            print(f"skipping {path}. I don't do word files!")
            return
        with open(path) as f:
            file_code = f.read()
        name = os.path.split(path)[1]
        total_text += make_file_header(name, font_name, font_size, page_width, page_height)
        total_text += "\n{file_code}\n"

    efipy.run(handle_file, input_path, files_filter="*", b_recursive=True)

    # cap line width
    total_text = total_text.replace("\t", "   ")
    lines = total_text.split("\n")
    lines_caped = []
    pages = []
    for line in lines:
        while True:
            lines_caped.append(line[:page_width])
            if len(line) < page_width:
                break
            line = line[page_width:]

    lines_caped_copy = copy.copy(lines_caped)
    # cap page height
    while len(lines_caped_copy) > 0:
        pages.append(lines_caped_copy[:page_height])
        lines_caped_copy = lines_caped_copy[page_height:]

    # add frame to pages
    if b_has_frame:
        corner_char = "O"
        frame_char = "X"
    else:
        corner_char = ""
        frame_char = ""
    for page in pages:
        put_page_in_frame(page, page_width, page_height, corner_char, frame_char)

    final_out = ""
    for page in pages:
        final_out += "\n".join(page)
        final_out += "\f"

    with open(output_path, "w+") as f:
        f.write(final_out)

    # open temp file in file editor:
    file_editors = ["start notepad++ ", "noteped ", "gedit ", ""]
    b_worked = None
    for fe in file_editors:
        b_worked = os.system(f"{fe}{output_path}")
        if b_worked == 0:
            break
    if b_worked != 0:
        print("something went wrong, couldn't open file editor.")
    print(f"the output file can be reached at the following path:\n{os.path.join(os.path.basename(__file__), output_path)}")
    print("\n\n\t\tpress enter to exit...")

    input()
