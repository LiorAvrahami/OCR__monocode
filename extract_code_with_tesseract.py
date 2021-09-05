import os

def process_image_to_text_with_tesseract(path):
    os.system(f"tesseract \"{path}\" tesseract_letter_detector\temp  nobatch tesseract_letter_detector\letters.txt > tesseract_letter_detector\log.txt")
    with open("temp.txt") as f:
        return f.read()