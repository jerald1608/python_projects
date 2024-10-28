from ultralytics import YOLO
from gtts import gTTS
import gradio as gr
import os

model = YOLO("yolov8m.pt")

def list_to_word_count_dict(word_list):
    word_count_dict = {}
    for word in word_list:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
    return word_count_dict
  
def word_count_dict_to_list(word_count_dict):
    word_list = []
    for word, count in word_count_dict.items():
        word_list.append(f"{count} {word}")
    return word_list

def process_image(img):
    # Perform YOLO prediction
    results = model.predict(img)
    result = results[0]

    # Initialize an empty text message
    text_message = []

    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        text_message.append(label)

    # Convert the list of labels to a word count dictionary
    word_count_dict = list_to_word_count_dict(text_message)

    # Convert the word count dictionary back to a list
    new_word_list = word_count_dict_to_list(word_count_dict)

    # Generate the final text message
    if len(new_word_list) == 0:
        final_message = "No objects detected in front of you."
    elif len(new_word_list) == 1:
        final_message = f"There is {new_word_list[0]} in front of you."
    else:
        final_message = "There are " + ", ".join(new_word_list[:-1]) + " and " + new_word_list[-1] + " in front of you."

    # Convert the text message to speech
    tts = gTTS(final_message)

    # Save the audio file to a specific path
    output_file_path = os.path.join(os.getcwd(), "output.mp3")  # Save in the current working directory
    try:
        tts.save(output_file_path)
        return output_file_path
    except Exception as e:
        print(f"Error saving TTS file: {e}")
        return None  # Return None or a suitable error message

iface = gr.Interface(fn=process_image, inputs="image", outputs="audio")
iface.launch(share=True)  # Set share=True to create a public link
