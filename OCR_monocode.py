import argparse
from efipy import run, inquire_output_path,inquire_input_path
from cross_correlation_letter_detector.handle_font_book_keepeing import inquire_font

import os
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
user_arguments.exportfilesystem = str_to_bool(user_arguments.exportfilesystem)
user_arguments.frame = str_to_bool(user_arguments.frame)
user_arguments.usetesseract = str_to_bool(user_arguments.usetesseract)

### PROCESS USER INPUT

# inquire input and output path if not given
input_path = inquire_input_path("") if user_arguments.inputpath is None else user_arguments.inputpath
errors_log_file_path = os.path.join(input_path,"OCR_errors.txt") if os.path.isdir(input_path) else os.path.join(os.path.dirname(input_path),"OCR_errors.txt")

print(os.path.isdir(input_path))

if os.path.isdir(input_path) or ((not os.path.isfile(input_path)) and "." not in os.path.basename(input_path)):
    default_output = os.path.join(input_path,"OCR_monocode_OUT")
elif os.path.isfile(input_path) or ((not os.path.isdir(input_path)) and "." in os.path.basename(input_path)):
    default_output = os.path.join(os.path.dirname(input_path),"OCR_monocode_OUT")
else:
    raise ImportError("it should had been impossible to reach this code. some random number:1684654987135468")

output_path = inquire_output_path(default_output) if user_arguments.outputpath is None else user_arguments.outputpath

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
out_strings = {}
def process_image_to_text(path):
    global out_strings
    if ocr_Mechanism == "ccr":
        out_strings[path] = process_image_to_text_with_ccr(path,selected_font_settings,b_has_frame= bool(user_arguments.frame))
    if ocr_Mechanism == "tesseract":
        out_strings[path] = process_image_to_text_with_tesseract(path)

run(process_image_to_text,root_path=input_path,files_filter="*.jpg",number_of_threads=3,errors_log_file=errors_log_file_path)

### combine files strings into one long string
out_str = ""
out_str_with_page_names = ""
for path in sorted(out_strings.keys()):
    out_str += out_strings[path]
    if out_str[-1] != "\n":
        out_str += "\n"
    out_str_with_page_names += f"-- %%% --\n" \
                               f"-- %%% -- new page. path : {path}\n" \
                               f"-- %%% --\n"
    out_str_with_page_names += out_strings[path]
    if out_str_with_page_names[-1] != "\n":
        out_str_with_page_names += "\n"

### make output path
if os.path.isfile(output_path):
    output_dir = os.path.dirname(output_path)
else:
    output_dir = output_path

def make_dir_recursive(path,depth=10):
    try:
        os.mkdir(path)
    except:
        if depth > 0:
            make_dir_recursive(os.path.dirname(path), depth=depth - 1)
            os.mkdir(path)
        else:
            raise Exception("path too deep")

if not os.path.exists(output_dir):
    try:
        make_dir_recursive(output_dir)
    except:
        raise AttributeError("output_path too deep")

### save combined string to file
if os.path.isdir(output_path):
    output_str_file = os.path.join(output_path,"raw_string_output.txt")
else:
    output_str_file = output_path
with open(output_str_file, "w+") as f:
    f.write(out_str_with_page_names)

### PROCESS OCR RESULTS AND WRITE TO FILE
# fix small chars errors
for ch in [" _","  , "," ."," `","   -   "]:
    out_str = re.sub(re.escape(ch)," "*len(ch),out_str)
out_str = re.sub(" *\n","\n",out_str)


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

for file_name, content in files:
    target_path = os.path.join(output_dir, file_name)
    if not os.path.exists(os.path.dirname(target_path)):
        make_dir_recursive(os.path.dirname(target_path))
    with open(target_path, "w+") as f:
        f.write(content)