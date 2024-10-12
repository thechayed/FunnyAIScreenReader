import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import json
import time
from ScreenReader import *

class GreetingWindow:
    def __init__(self, master):
        # The master window for the GreetingWindow
        self.master = master
        # The title of the GreetingWindow
        self.master.title("Welcome to Screen Reader")
        # The size of the GreetingWindow
        self.master.geometry("600x400")
        # Method to create the widgets for the GreetingWindow
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets for the GreetingWindow, which is a window that
        pops up when the application is first opened. It contains a welcome
        message and a button to launch the main application. The window is
        centered on the screen and is 600x400 pixels in size.

        :return: None
        """

        frame = ttk.Frame(self.master, padding="20")
        frame.pack(expand=True, fill='both')

        ttk.Label(frame, text="Welcome to the Screen Reader Application!", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(frame, text="This app analyzes your screen and provides real-time commentary.").pack(pady=5)
        ttk.Label(frame, text="Click the button below to start.").pack(pady=10)

        ttk.Button(frame, text="Launch Main Application", command=self.open_main_app).pack(pady=20)

    def open_main_app(self):
        """
        Closes the GreetingWindow and opens the main application window.

        The method is called when the user clicks the "Launch Main Application" button in the GreetingWindow.

        :return: None
        """
        self.master.destroy()  # Close the greeting window
        root = tk.Tk()
        app = TkinterApp(root, Config())
        root.mainloop()

class TkinterApp:
    """
    The TkinterApp class is a graphical user interface (GUI) application built using the Tkinter library. It provides a screen reader functionality that can extract and generate descriptions of the current screen.
    """
    def __init__(self, master, config):

        self.master = master
        # The configuration for the ScreenAnalyzer
        self.config = config
        # The ScreenAnalyzer object
        self.analyzer = ScreenAnalyzer(config)
        # Flag to indicate if the screen reader is currently active
        self.screen_reader_active = False
        # The thread for the screen reader
        self.screen_reader_thread = None

        self.master.title("Screen Reader Application")
        self.master.geometry("800x600")
        
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.create_widgets()

    def create_widgets(self):
        # Create notebook for tabs
        """
        Creates all the widgets for the application, including the notebook, frames, scrolled text box, listbox, and buttons.

        :return: None
        """
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.grid(row=0, column=0, sticky='nsew')

        # Create frames for each tab
        self.cached_results_frame = ttk.Frame(self.notebook)
        self.chatbot_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.cached_results_frame, text='Cached Results')
        self.notebook.add(self.chatbot_frame, text='Chatbot')

        # Configure weights for frames
        self.cached_results_frame.grid_rowconfigure(0, weight=1)
        self.cached_results_frame.grid_columnconfigure(0, weight=1)
        self.chatbot_frame.grid_rowconfigure(0, weight=1)
        self.chatbot_frame.grid_columnconfigure(0, weight=1)

        # Cached Results Tab
        self.cached_results_text = scrolledtext.ScrolledText(self.cached_results_frame, wrap=tk.WORD)
        self.cached_results_text.grid(row=0, column=0, sticky='nsew')

        # Chatbot Tab
        self.chatbot_list = tk.Listbox(self.chatbot_frame)
        self.chatbot_list.grid(row=0, column=0, sticky='nsew')

        # Control Panel
        control_frame = ttk.Frame(self.master)
        control_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=10)
        control_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Labels
        ttk.Label(control_frame, text="Screen Reader Status:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.status_label = ttk.Label(control_frame, text="OFF")
        self.status_label.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(control_frame, text="Last Update:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.last_update_label = ttk.Label(control_frame, text="N/A")
        self.last_update_label.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5)
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.toggle_button = ttk.Button(button_frame, text="Start Screen Reader", command=self.toggle_screen_reader)
        self.toggle_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.update_cache_button = ttk.Button(button_frame, text="Update Cache", command=self.update_cache)
        self.update_cache_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.clear_chat_button = ttk.Button(button_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_chat_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')

    def toggle_screen_reader(self):
        """
        Toggles the screen reader functionality on and off.

        If the screen reader is currently running, it will be stopped and the
        status label will be updated to "OFF". The toggle button text will be
        updated to "Start Screen Reader".

        If the screen reader is currently not running, it will be started and
        the status label will be updated to "ON". The toggle button text will
        be updated to "Stop Screen Reader". The screen reader thread will be
        started.

        :return: None
        """
        if self.screen_reader_active:
            self.screen_reader_active = False
            self.status_label.config(text="OFF")
            self.toggle_button.config(text="Start Screen Reader")
        else:
            self.screen_reader_active = True
            self.status_label.config(text="ON")
            self.toggle_button.config(text="Stop Screen Reader")
            self.screen_reader_thread = threading.Thread(target=self.run_screen_reader)
            self.screen_reader_thread.start()

    def run_screen_reader(self):
        """
        Runs the screen reader thread.

        This function is intended to be run as a separate thread, and will
        continue running until the `screen_reader_active` flag is set to
        False. It will call the `extract_and_generate_description` method
        of the `analyzer` to generate a description of the current screen,
        and will then call the `update_chatbot` method to update the chatbot
        list with the description. The `update_last_update_label` method is
        also called to update the last update label.

        :return: None
        """
        while self.screen_reader_active:
            description = self.analyzer.extract_and_generate_description()
            self.master.after(0, self.update_chatbot, description)
            self.master.after(0, self.update_last_update_label)
            time.sleep(5)

    def update_chatbot(self, message):
        """
        Updates the chatbot listbox with a new message.

        :param message: The message to be added to the chatbot listbox
        :return: None
        """
        self.chatbot_list.insert(tk.END, message)
        self.chatbot_list.see(tk.END)

    def update_last_update_label(self):
        """
        Updates the last update label with the current time.

        :return: None
        """
        current_time = time.strftime("%H:%M:%S")
        self.last_update_label.config(text=current_time)

    def update_cache(self):
        """
        Updates the cached results text box with the current responses stored in the cache.

        This function is intended to be called when the "Update Cache" button is clicked.
        It will load the current responses from the cache, clear the text box, and then
        insert the responses into the text box. The responses will be inserted in the
        format "Input: <key>\nResponse: <value>\n\n", where <key> is the input string
        and <value> is the response string.

        :return: None
        """
        responses = self.analyzer.response_manager.load_responses()
        self.cached_results_text.delete('1.0', tk.END)
        for key, value in responses.items():
            self.cached_results_text.insert(tk.END, f"Input: {key}\nResponse: {value}\n\n")

    def clear_chat(self):
        """
        Clears the chatbot listbox of all messages.

        This function is intended to be called when the "Clear Chat" button is clicked.
        It will delete all messages from the chatbot listbox.

        :return: None
        """
        self.chatbot_list.delete(0, tk.END)


def main():
    greeting_root = tk.Tk()
    greeting_app = GreetingWindow(greeting_root)
    greeting_root.mainloop()

if __name__ == "__main__":
    main()