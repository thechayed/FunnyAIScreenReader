import pyautogui
import pytesseract
import openai
import time
from google.cloud import vision
import os
import base64
import random
import io
import gzip
import json
import numpy as np
from sklearn.cluster import KMeans
import requests
import inspect


class Config:
    def __init__(self):
        keys_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/Keys/"
        f = open(keys_path + 'openaikey.txt', 'r')
        self.openai_api_key = f.read()
        self.google_credentials_path = keys_path + "Keys/brave-reason-435220-m5-65204904b50a.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_credentials_path
        self.tesseract_cmd_path = r'Tesseract-OCR\tesseract.exe'
        
        
        self.response_file = 'responses.json'
        self.ORIGINAL_SIMILARITY_THRESHOLD = 1.02
        self.similarity_threshold = 0.9
        self.similarity_threshold_depletion_rate = 0.001
        self.similarity_randomness = 0.05
        self.storage_threshold = 1.01
        self.vision_api_interval = 5  # Run Vision API every 5 iterations
        self.vision_api_counter = 0  # Counter to
        self.messages_file = "messages.json"

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd_path

    def capture_screen(self, compression_quality=50):
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format='JPEG', quality=compression_quality)
        return screenshot

    def extract_text_from_image(self, image):
        return pytesseract.image_to_string(image)

    def save_texture(self, image, filename="last_capture.png"):
        image.save(filename)
        print(f"Texture saved as {filename}")

    def analyze_image(self, image):
        image_bytes = image.tobytes()
        self.save_texture(image)
        vision_image = vision.Image(content=image_bytes)
        response = self.vision_client.label_detection(image=vision_image)
        labels = response.label_annotations
        label_descriptions = [label.description for label in labels]
        return ", ".join(label_descriptions)

class ResponseManager:
    def __init__(self, config):
        self.config = config

    def load_responses(self):
        # try:
            f = open(self.config.response_file, 'r+')
            content = f.read()
            if not content:
                print("The file is empty")
                responses = {}
                f.write(json.dumps(responses))
                return responses
            responses = json.loads(content)
            return responses
        # except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        #     print(f"Error loading responses: {e}")
        #     return {}

    def save_responses(self, responses):
        cleaned_responses = {str(k): self.bytes_to_string(v) for k, v in responses.items()}
        with open(self.config.response_file, 'w') as f:
            json.dump(cleaned_responses, f)

    def store_response(self, input_data, response):
        responses = self.load_responses()
        compressed_input = self.gzip_compress(input_data)
        responses[compressed_input] = response
        self.save_responses(responses)

    @staticmethod
    def bytes_to_string(obj):
        return obj.decode('utf-8') if isinstance(obj, bytes) else obj

    @staticmethod
    def gzip_compress(data):
        return gzip.compress(data.encode('utf-8'))

    @staticmethod
    def gzip_decompress(data):
        return gzip.decompress(data).decode('utf-8')

class ResponseMatcher:
    def __init__(self, response_manager):
        self.response_manager = response_manager

    def gzip_distance(self, s1, s2):
        c1 = len(gzip.compress(s1.encode()))
        c2 = len(gzip.compress(s2.encode()))
        c12 = len(gzip.compress((s1 + s2).encode()))
        return (c12 - min(c1, c2)) / max(c1, c2)

    def find_closest_match(self, input_data):
        responses = self.response_manager.load_responses()
        if len(responses) == 0:
            return None

        input_keys = list(responses.keys())
        response_values = list(responses.values())

        training_vectors = np.array([len(gzip.compress(k.encode())) for k in input_keys]).reshape(-1, 1)
        input_vector = np.array([len(gzip.compress(input_data.encode()))]).reshape(1, -1)

        n_clusters = min(len(training_vectors), 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random.randint(0, 2)).fit(training_vectors)

        closest_cluster = kmeans.predict(input_vector)[0]
        cluster_indices = np.where(kmeans.labels_ == closest_cluster)[0]

        closest_index = min(cluster_indices, key=lambda i: self.gzip_distance(input_data, input_keys[i]))

        closest_input_key = input_keys[closest_index]
        closest_response = responses[closest_input_key]
        distance = self.gzip_distance(input_data, closest_input_key)

        return closest_input_key, closest_response, distance

class GPTHandler:
    def __init__(self, config):
        self.config = config
        openai.api_key = config.openai_api_key
        self.previous_message = ""

    def generate_description(self, input_data):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You speak in one or two sentences, you are a mean slug named Murph. You are a goofy, grumpy and rude slug buddy. Sometimes, you will give specific tips about the contents on the screen. If the user seems to have anything fun open you make funny comments."},
                {"role": "user", "content": f"don't repeat yourself, this is what you said before, so avoid discussing the same topic: {self.previous_message}, Make a comment to the user who's screen displays this content (this is a pytesseract generated OCR string, comment on one specific part of the content, focus on what the user is most likely to be looking at. The image labels are from Google Vision API): {input_data} "}
            ]
        )
        self.previous_message = response.choices[0].message.content
        return self.previous_message

    def generate_vision_summary(self, image):
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openai_api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a tutor slug friend, you should briefly explain what you see then create todo bulletins of correction or improvement suggestions based on the activity performed on the screen cap you see. Throw in some crude humor."},
                {
                "role": "user",
                "content": [
                        {
                        "type": "text",
                        "text": "What needs to be fixed in this image?"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        return response.content

class ScreenAnalyzer:
    def __init__(self, config):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.response_manager = ResponseManager(config)
        self.response_matcher = ResponseMatcher(self.response_manager)
        self.gpt_handler = GPTHandler(config)
        self.last_vision_summary = ""

    def extract_and_generate_description(self):
        screenshot = self.image_processor.capture_screen()
        extracted_text = self.image_processor.extract_text_from_image(screenshot)
        # image_analysis = self.image_processor.analyze_image(screenshot)
        
        self.last_vision_summary = ""
        # if self.config.vision_api_counter % self.config.vision_api_interval == 0:
        #     self.last_vision_summary = self.gpt_handler.generate_vision_summary(screenshot)
        self.config.vision_api_counter += 1

        input_data = f"pytesseract string: {extracted_text}, vision summary: {self.last_vision_summary}"
        match = self.response_matcher.find_closest_match(input_data)

        if match and match[2] < self.config.similarity_threshold:
            self.config.similarity_threshold -= self.config.similarity_threshold_depletion_rate
            return match[1]

        description = self.gpt_handler.generate_description(input_data)
        
        if match:
            if match[2] > self.config.storage_threshold:
                self.response_manager.store_response(input_data, description)
        else:
            self.response_manager.store_response(input_data, description)
        
        self.config.similarity_threshold = self.config.ORIGINAL_SIMILARITY_THRESHOLD
        
        return description

def main():
    config = Config()
    analyzer = ScreenAnalyzer(config)
    
    messages_file = open(config.messages_file, "w")
    messages = []

    while True:
        description = analyzer.extract_and_generate_description()
        print("Murph:")
        print(description)
        print("\n" + "-"*50 + "\n")
        
        messages.append(description)
        messages_file.write(json.dumps(messages))
        
        config.similarity_threshold -= config.similarity_threshold_depletion_rate
        time.sleep(5)
        

if __name__ == "__main__":
    main()