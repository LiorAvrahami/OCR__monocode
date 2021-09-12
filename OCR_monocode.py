import argparse
from efipy import run, inquire_output_path,inquire_input_path
from cross_correlation_letter_detector.handle_font_book_keepeing import inquire_font

import os,matplotlib.pyplot as plt
from extract_code_with_crosscorrelation import process_image_to_text_with_ccr
from extract_code_with_tesseract import process_image_to_text_with_tesseract
import re

def str_to_bool(s):
    if str.lower(s) in ["true", "yes"]:
        return True
    if str.lower(s) in ["false", "no"]:
        return False
    raise AttributeError(f"unrecognized value {s}, expected true or false.")

argparser = argparse.ArgumentParser(description="OCR program, that reads text from image and exports to text file.\nsee https://github.com/LiorAvrahami/OCR__monocode")
argparser.add_argument("--exportfilesystem",action='store',nargs='?', default="false", const="true", choices=["true","false"],
                       help="if this atribute is added then no OCR will be used, intead, the selected input file or folder that represents the root of a file system, will be converted to a single .txt file that can be printed, and can later be scanned and mapped back to the origional file system hyrarchy")
argparser.add_argument("--singletextfileoutput",action='store',nargs='?', default="false", const="true", choices=["true","false"],
                       help="if selected output will be a single .txt file, no file system interpretation will be done. (so when using this flag there is no limmit on use of $ char)")
argparser.add_argument("-i", "--inputpath",action='store', default=None,
                       help="select input path. may be a folder, or path to single file")
argparser.add_argument("-o", "--outputpath",action='store', default=None,
                       help="select output path. may be a folder, or path to single file")
argparser.add_argument("-f", "--fontselect", action='store', nargs=3, default=None,
                       help="select font specifics [font name] [font size] [scanner resolution]dpi\n   example:\n      python OCR_monocode.py -fs Consols 8 600dpi")
argparser.add_argument("-n", "--hasnoframe", action='store',nargs='?', default="true", const="false", choices=["true","false"], dest="frame",
                       help="there is an option to surround the paper with a frame of X, to increase percision. if using \"--exportfilesystem\" then the correct frame will be automatically added to the text files. this frame will not be in the final .txt files.")
argparser.add_argument("-t", "--usetesseract", action='store', nargs='?', default="false", const="true", choices=["true","false"],
                       help="use tesseract instead of the cross correlation ccr developed in this project.")

user_arguments = argparser.parse_args()
print(user_arguments)
user_arguments.exportfilesystem = str_to_bool(user_arguments.exportfilesystem)
user_arguments.singletextfileoutput = str_to_bool(user_arguments.singletextfileoutput)
user_arguments.frame = str_to_bool(user_arguments.frame)
user_arguments.usetesseract = str_to_bool(user_arguments.usetesseract)

### PROCESS USER INPUT

# inquire input and output path if not given
input_path = inquire_input_path(".") if user_arguments.inputpath is None else user_arguments.inputpath
output_path = inquire_output_path(os.path.join(input_path,"OCR_monocode_OUT")) if user_arguments.outputpath is None else user_arguments.outputpath

# if user wants to export some of their file system
if user_arguments.exportfilesystem:
    from format_files_to_be_printed import code_to_print
    code_to_print.run(input_path,output_path,b_has_frame = bool(user_arguments.frame))
    quit(0)

# note selected ocr mechanism
if bool(user_arguments.usetesseract):
    ocr_Mechanism = "tesseract"
else:
    ocr_Mechanism = "ccr"

# inquire font settings if not given
if user_arguments.fontselect is not None:
    selected_font_settings = user_arguments.fontselect
else:
    selected_font_settings = inquire_font()


### START RUNNING OCR
out_str = ""
def process_image_to_text(path):
    global out_str
    if ocr_Mechanism == "ccr":
        out_str += process_image_to_text_with_ccr(path,selected_font_settings,b_has_frame= bool(user_arguments.frame))
    if ocr_Mechanism == "tesseract":
        out_str += process_image_to_text_with_tesseract(path)
    if out_str[-1] != "\n":
        out_str += "\n"

run(process_image_to_text,root_path=input_path,files_filter="*.jpg")


### PROCESS OCR RESULTS AND WRITE TO FILE
# fix small chars errors
for ch in [" _","  , "," ."," `","   -   "]:
    out_str = re.sub(re.escape(ch)," "*len(ch),out_str)
out_str = re.sub(" *\n","\n",out_str)

if not bool(user_arguments.singletextfileoutput):
    # split text into file system
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

    if os.path.isfile(output_path):
        output_path = os.path.dirname(output_path)

    if not os.path.exists(output_path):
        try:
            os.mkdir(output_path)
        except:
            raise AttributeError("output_path too deep")

    for file_name, content in files:
        with open(os.path.join(output_path,file_name), "w+") as f:
            f.write(content)
else:
    # write text to file
    if not os.path.isfile(output_path):
        output_path = os.path.join(output_path,"out.txt")
    with open(output_path, "w+") as f:
        f.write(out_str)