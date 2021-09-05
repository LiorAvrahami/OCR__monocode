from cross_correlation_letter_detector import main

def process_image_to_text_with_ccr(path):
    score,text = main(path)
    print(f"score for {path} is {score:.3g}")
    return text