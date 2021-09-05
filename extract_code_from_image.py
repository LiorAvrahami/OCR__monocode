from efipy import run
import os,matplotlib.pyplot as plt
from extract_code_with_crosscorrelation import process_image_to_text_with_ccr
from extract_code_with_tesseract import process_image_to_text_with_tesseract
import re

ocr_Mechanism = "ccr"

out_str = ""
def process_image_to_text(path):
    global out_str
    if ocr_Mechanism == "ccr":
        out_str += process_image_to_text_with_ccr(path)
    if ocr_Mechanism == "tesseract":
        out_str += process_image_to_text_with_tesseract(path)
    if out_str[-1] != "\n":
        out_str += "\n"

run(process_image_to_text,files_filter="*.jpg",b_inquire_output=True)

for ch in [" _","  , "," ."," `","   -   "]:
    out_str = re.sub(re.escape(ch)," "*len(ch),out_str)
out_str = re.sub(" *\n","\n",out_str)

lines = out_str.split("\n")
for i in range(len(lines)):
    if len(lines[i]) > 120 and i < len(lines) - 1:
        lines[i] += lines[i + 1]
        lines[i + 1] = ""
lines.append("$$q$$")
files = []
cur_file_name = None
cur_file_start = 0
for i in range(len(lines)):
    if lines[i][:2] == "$$":
        #eof found
        if cur_file_name is not None:
            files.append((cur_file_name,"\n".join(lines[cur_file_start:i])))
        cur_file_start = i + 1
        cur_file_name = re.match("\$\$(.+?)\$\$",lines[i]).groups()[0]

if not os.path.exists("results"):
    os.mkdir("results")

for file_name,content in files:
    with open(f"results\\{file_name}", "w+") as f:
        f.write(content)

# print(out_str)