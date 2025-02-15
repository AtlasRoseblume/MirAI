import customtkinter as ctk
import os
import random
from time import sleep, time
from threading import Thread
from PIL import Image

class UI:
    def __init__(self, core, image_path: str = "images", title: str = "MirAI"):
        self.running = True
    
        self.background_thread = Thread(target=UI.run_customtkinter, args=(self, core, image_path, title))
        self.background_thread.start()

    
    def load_images(self, folder_path, image_size):
        images = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(".webp"): # TODO: Support other image formats
                try:
                    img = Image.open(os.path.join(folder_path, file))
                    img = img.resize(image_size, Image.Resampling.LANCZOS)
                    images.append(ctk.CTkImage(light_image=img, size=image_size))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        return images

    def close(self):
        self.exit_app()
        self.background_thread.join()

    def exit_app(self):
        self.running = False

    @staticmethod
    def get_history(history: str, limit=70):
        history = history.rstrip()

        if len(history) <= limit:
            return history
        
        substring = history[-limit:]

        space_index = substring.find(' ')

        if space_index != -1:
            return substring[space_index:]
        else:
            return substring

    def run_customtkinter(self, core, image_path, title):
        # ctk.set_default_color_theme("blue")

        app = ctk.CTk()
        app.title(title)
        app.attributes('-fullscreen', True)
        app.configure(fg_color="#1F1F1F") 
        self.images = self.load_images(image_path, (340, 340))
 
        exit_button = ctk.CTkButton(
            app,
            text="Exit",
            command=lambda: self.exit_app(),
            corner_radius=10,
            font=("Arial", 16, "bold")
        )
        exit_button.pack(pady=10, padx=20, anchor='se', side='bottom')

        cheat_button = ctk.CTkButton(
            app, 
            text="Cheat Mode", 
            command=lambda: core.toggle_cheat_mode(),
            corner_radius=10,
            font=("Arial", 16, "bold"),
            fg_color="red"
        )
        cheat_button.pack(padx=20, anchor='se', side='bottom')

        status_label = ctk.CTkLabel(
            app,
            text = "Started!",
            font=("Arial", 20, "bold"),
            text_color="white"    
        )

        sublabel_1 = ctk.CTkLabel(
            app,
            text="",
            font=("Arial", 20, "bold"),
            text_color="#65FF65"
        )

        sublabel_1.pack(pady=0, anchor='s', side='bottom')
        status_label.pack(pady=0, anchor='s', side='bottom')
        
        label = ctk.CTkLabel(app, text="")
        label.place(relx=0.5, rely=0.375, anchor="center")
        new_img = random.choice(self.images)
        label.configure(image=new_img)

        min_interval = 10
        max_interval = 20

        countdown = random.uniform(min_interval, max_interval)
        start_time = time()
        curr_time = time()
        while self.running:
            try:
                app.update_idletasks()
                app.update()

                curr_time = time()
                if (curr_time - start_time) > countdown:
                    new_img = random.choice(self.images)
                    label.configure(image=new_img)
                    start_time = curr_time
                    countdown = random.uniform(min_interval, max_interval)

                if core.model.cheat_mode:
                    cheat_button.configure(fg_color="green")
                else:
                    cheat_button.configure(fg_color="red")

                if core.listening:
                    status_label.configure(text="Listening...")
                    sublabel_1.configure(text=UI.get_history(core.buffer))
                else:
                    status_label.configure(text="Thinking...")
                    if len(core.response_buffer) == 0:
                        sublabel_1.configure(text=UI.get_history(core.captured_text))
                    else:
                        sublabel_1.configure(text=UI.get_history(core.response_buffer))

                sleep(0.01)
            except:
                self.running = False
                break
        
        app.destroy()
