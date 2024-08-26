from google.cloud import translate_v2 as translate

def detect_language_and_translate(text):
    # Initialize the Translate API client
    translate_client = translate.Client()

    # Detect the language
    detection = translate_client.detect_language(text)
    detected_language = detection['language']

    print(f"Detected language: {detected_language}")

    # Translate the text to English
    translation = translate_client.translate(text, target_language='en')

    print(f"Translated text: {translation['translatedText']}")

    return detected_language, translation['translatedText']

# Example usage
text = "Hola, ¿cómo estás?"
detected_language, translated_text = detect_language_and_translate(text)

print(f"Original Text: {text}")
print(f"Detected Language: {detected_language}")
print(f"Translated Text: {translated_text}")