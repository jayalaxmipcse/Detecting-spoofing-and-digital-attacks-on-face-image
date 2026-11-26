
import pyttsx3

print("Starting voice test...")

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')
print("Available voices:", len(voices))

engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

engine.say("Hello! Testing voice alert.")
engine.runAndWait()

print("Voice test completed!")
