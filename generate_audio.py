import pyttsx3

def generate(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, "manual.mp3")
    engine.runAndWait()

if __name__ == "__main__":
    text = ""
    with open("manual_guide.txt", "r", encoding='utf-8') as f:
        text = f.read()
    generate(text)