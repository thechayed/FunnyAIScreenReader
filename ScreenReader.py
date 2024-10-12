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
    """
    Configuration class for the ScreenReader class.
    """
    def __init__(self):
        # Set the path to the Keys folder relative to the current file
        keys_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/Keys/"
        
        # OpenAI API Key: Read from a file named 'openaikey.txt'
        f = open(keys_path + 'openaikey.txt', 'r')  
        self.openai_api_key = f.read()  # Store the OpenAI API key
        
        # Google API Key: Set the path to the Google credentials JSON file
        self.google_credentials_path = keys_path + "Keys/brave-reason-435220-m5-65204904b50a.json"
        
        # Set the environment variable for Google Cloud authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_credentials_path
        
        # Path to the Tesseract OCR executable
        self.tesseract_cmd_path = r'Tesseract-OCR\tesseract.exe'
        
        # File name for storing responses
        self.response_file = 'responses.json'
        
        # Original similarity threshold for response matching
        self.ORIGINAL_SIMILARITY_THRESHOLD = 1.02
        
        # Current similarity threshold for response matching
        self.similarity_threshold = 0.9
        
        # Rate at which the similarity threshold decreases
        self.similarity_threshold_depletion_rate = 0.001
        
        # Randomness factor to introduce variability in similarity matching
        self.similarity_randomness = 0.05
        
        # Threshold to determine if a response is stored based on similarity
        self.storage_threshold = 1.01
        
        # Interval for running the Vision API (every 5 iterations)
        self.vision_api_interval = 5  
        
        # Counter to keep track of how many times the Vision API has been run
        self.vision_api_counter = 0  
        
        # File name for storing messages
        self.messages_file = "messages.json"

class ImageProcessor:
    def __init__(self, config):
        """
        Initialize the ImageProcessor class with a Config object.

        Parameters:
            config (Config): configuration containing the path to the Tesseract OCR executable

        Returns:
            None
        """
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd_path

    def capture_screen(self, compression_quality=50):
        """
        Capture a screenshot of the current screen and return it as a PIL Image object.

        Parameters:
            compression_quality (int): quality of the JPEG image, from 1 (worst) to 95 (best).
                Defaults to 50.

        Returns:
            PIL.Image: screenshot as a PIL Image object
        """
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format='JPEG', quality=compression_quality)
        return screenshot

    def extract_text_from_image(self, image):
        """
        Extract text from the given image using Tesseract OCR.

        Parameters:
            image (PIL.Image): image to extract text from

        Returns:
            str: extracted text
        """
        return pytesseract.image_to_string(image)

    def save_texture(self, image, filename="last_capture.png"):
        """
        Save the given image as a PNG file to the disk.

        Parameters:
            image (PIL.Image): image to save
            filename (str): filename to save the image as, defaults to "last_capture.png"

        Returns:
            None
        """
        image.save(filename)
        print(f"Texture saved as {filename}")

    def analyze_image(self, image):
        """
        Analyze the given image using Google Cloud Vision API and return a string describing it.

        Parameters:
            image (PIL.Image): image to analyze

        Returns:
            str: description of the image
        """
        image_bytes = image.tobytes()
        self.save_texture(image)
        vision_image = vision.Image(content=image_bytes)
        response = self.vision_client.label_detection(image=vision_image)
        labels = response.label_annotations
        label_descriptions = [label.description for label in labels]
        return ", ".join(label_descriptions)

class ResponseManager:
    """
    A class to manage storing and retrieving responses from the disk.

    Attributes:
        config (Config): the configuration object that contains the path to the file where responses are stored
    """

    def __init__(self, config):
        """
        Initialize the ResponseManager with the given configuration object

        Parameters:
            config (Config): the configuration object
        """
        self.config = config

    def load_responses(self):
        """
        Load the responses from the file specified in the configuration object

        Returns:
            dict: a dictionary mapping input strings to their corresponding responses. If the file is empty, an empty dictionary is returned.

        Raises:
            json.JSONDecodeError: if the file does not contain valid JSON
        """
        try:
            with open(self.config.response_file, 'r+') as f:
                content = f.read()
                if not content:
                    print("The file is empty")
                    responses = {}
                    f.write(json.dumps(responses))
                    return responses
                responses = json.loads(content)
                return responses
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            print(f"Error loading responses: {e}")
            return {}

    def save_responses(self, responses):
        """
        Save the given responses to the file specified in the configuration object

        Parameters:
            responses (dict): a dictionary mapping input strings to their corresponding responses
        """
        cleaned_responses = {str(k): self.bytes_to_string(v) for k, v in responses.items()}
        with open(self.config.response_file, 'w') as f:
            json.dump(cleaned_responses, f)

    def store_response(self, input_data, response):
        """
        Store the given response for the given input_data

        Parameters:
            input_data (str): the input string
            response (str): the response string
        """
        responses = self.load_responses()
        compressed_input = self.gzip_compress(input_data)
        responses[compressed_input] = response
        self.save_responses(responses)

    @staticmethod
    def bytes_to_string(obj):
        """
        Convert the given object to a string if it is a bytes object

        Parameters:
            obj (bytes): the object to convert

        Returns:
            str: the string representation of the object
        """
        return obj.decode('utf-8') if isinstance(obj, bytes) else obj

    @staticmethod
    def gzip_compress(data):
        """
        Compress the given data using gzip

        Parameters:
            data (str): the data to compress

        Returns:
            bytes: the compressed data
        """
        return gzip.compress(data.encode('utf-8'))

    @staticmethod
    def gzip_decompress(data):
        """
        Decompress the given data using gzip

        Parameters:
            data (bytes): the data to decompress

        Returns:
            str: the decompressed data
        """
        return gzip.decompress(data).decode('utf-8')

class ResponseMatcher:
    """
    This class is responsible for matching the given input with one of the stored responses.

    The matching algorithm is based on the gzip compression of the input and the stored responses.
    The idea is that strings that are most similar will have a similar compression size.
    """
    def __init__(self, response_manager):
        """
        Initialize the ResponseMatcher with the given response_manager.

        Parameters:
            response_manager (ResponseManager): the response manager to use
        """
        self.response_manager = response_manager

    def gzip_distance(self, s1, s2):
        """
        Calculate the distance between two strings based on their gzip compression.

        Parameters:
            s1 (str): the first string
            s2 (str): the second string

        Returns:
            float: the distance between the two strings
        """
        c1 = len(gzip.compress(s1.encode()))
        c2 = len(gzip.compress(s2.encode()))
        c12 = len(gzip.compress((s1 + s2).encode()))
        return (c12 - min(c1, c2)) / max(c1, c2)

    def find_closest_match(self, input_data):
        """
        Find the closest match for the given input_data among the stored responses.

        Parameters:
            input_data (str): the input string to match

        Returns:
            tuple: (closest_input_key, closest_response, distance)
        """
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
    """
    A class to generate descriptions using OpenAI's GPT models.
    """
    def __init__(self, config):
        """
        Initialize the GPTHandler with the given config.

        Parameters:
            config (Config): the configuration to use
        """
        # The configuration to use
        self.config = config
        # The OpenAI API key to use
        openai.api_key = config.openai_api_key
        # The previous message to avoid repeating it
        self.previous_message = ""

    def generate_description(self, input_data):
        """
        Generate a description based on the given input data.

        Parameters:
            input_data (str): the input string to generate a description for

        Returns:
            str: the generated description
        """
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
        """
        Generate a summary of the given image using OpenAI's GPT models.

        Parameters:
            image (PIL.Image): the image to generate a summary for

        Returns:
            str: the generated summary
        """
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        # Convert bytes to base64 string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openai_api_key}"
        }
        
        # Create payload for API request
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
        
        # Send API request and get response
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        return response.content

class ScreenAnalyzer:
    """
    A class to analyze the screen and generate descriptions for the user.
    """
    def __init__(self, config):
        """
        Initialize the ScreenAnalyzer with the given config.

        Parameters:
            config (Config): the configuration to use
        """
        self.config = config
        # ImageProcessor instance to process images
        self.image_processor = ImageProcessor(config)
        # ResponseManager instance to manage responses
        self.response_manager = ResponseManager(config)
        # ResponseMatcher instance to match responses
        self.response_matcher = ResponseMatcher(self.response_manager)
        # GPTHandler instance to generate descriptions
        self.gpt_handler = GPTHandler(config)
        # Last generated vision summary
        self.last_vision_summary = ""

    def extract_and_generate_description(self):
        """
        Extract the text from the screen and generate a description based on the extracted text.

        Returns:
            str: the generated description
        """
        screenshot = self.image_processor.capture_screen()
        extracted_text = self.image_processor.extract_text_from_image(screenshot)
        # image_analysis = self.image_processor.analyze_image(screenshot)
        
        # Generate vision summary every self.config.vision_api_interval times
        self.last_vision_summary = ""
        if self.config.vision_api_counter % self.config.vision_api_interval == 0:
            self.last_vision_summary = self.gpt_handler.generate_vision_summary(screenshot)
        self.config.vision_api_counter += 1

        # Create input data string for GPT
        input_data = f"pytesseract string: {extracted_text}, vision summary: {self.last_vision_summary}"

        # Find the closest match for the input data among the stored responses
        match = self.response_matcher.find_closest_match(input_data)

        # If a match is found and the distance is below the threshold, return the matched response
        if match and match[2] < self.config.similarity_threshold:
            self.config.similarity_threshold -= self.config.similarity_threshold_depletion_rate
            return match[1]

        # Generate a description using GPT
        description = self.gpt_handler.generate_description(input_data)
        
        # If a match is found and the distance is above the storage threshold, store the input data and the generated description
        if match:
            if match[2] > self.config.storage_threshold:
                self.response_manager.store_response(input_data, description)
        else:
            self.response_manager.store_response(input_data, description)
        
        # Reset the similarity threshold
        self.config.similarity_threshold = self.config.ORIGINAL_SIMILARITY_THRESHOLD
        
        return description

