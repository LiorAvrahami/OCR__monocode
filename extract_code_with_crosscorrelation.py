from cross_correlation_letter_detector import main

def process_image_to_text_with_ccr(path,selected_font_settings=None,b_has_frame=True):
    score,text = main(path,font=selected_font_settings,b_has_frame=b_has_frame)
    print(f"score for {path} is {score:.3g}")
    return text