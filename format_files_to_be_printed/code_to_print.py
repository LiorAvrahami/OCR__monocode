import os,copy,efipy
__version__ = "1.0"

excluded_file_types = [".docx",".wbk"]

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
    total_text += f"$${name}$$ Font:Consolas8 | PageWidth:{page_width} | PageHeight:{page_height}\n{file_code}\n"

efipy.run(handle_file,files_filter="*",b_recursive=True)

# cap line width
total_text = total_text.replace("\t","   ")
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
top_frame = "O" + "X"*page_width + "O"
bottom_frame = top_frame
right_frame_char = "X"
left_frame_char = "X"
for page in pages:
    page += [""]*(page_height - len(page))

    # right left bounds
    for i in range(len(page)):
        page[i] = left_frame_char + page[i] + " "*(page_width - len(page[i])) + right_frame_char

    # top bottom bounds
    page.insert(0,top_frame)
    page.append(bottom_frame)

final_out = "\f"
for page in pages:
    final_out += "\n".join(page)
    final_out += "\f"

out_file_name = "temp_out.txt"

with open(out_file_name,"w+") as f:
    f.write(final_out)

# open temp file in file editor:
file_editors = ["start notepad++ ","noteped ","gedit ",""]
b_worked = None
for fe in file_editors:
    b_worked = os.system(f"{fe}{out_file_name}")
    if b_worked == 0:
        break
if b_worked != 0:
    print("something went wrong, couldn't open file editor.")
print(f"the output file can be reached at the following path:\n{os.path.join(os.path.basename(__file__),out_file_name)}")
print("\n\n\t\tpress enter to exit...")

input()
