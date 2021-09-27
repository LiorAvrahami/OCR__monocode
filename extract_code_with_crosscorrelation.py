from cross_correlation_letter_detector import main

def process_image_to_text_with_ccr(path,selected_font_settings=None,b_has_frame=True,b_load_last_run=True):
    score,text = main(path,font=selected_font_settings,b_has_frame=b_has_frame,b_load_last_run=b_load_last_run)
    print(f"score for {path} is {score:.3g}")
    return text

if __name__ == "__main__":
    import efipy
    process_image_to_text_with_ccr(efipy.inquire_input_path(r"E:\Liors_Stuff\Programing M\python\from refael\p06-100003.jpg"),"consolas 8 600".split(" "),b_load_last_run=False)