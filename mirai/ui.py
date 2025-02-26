import customtkinter as ctk
import os
import random
import sys
from time import sleep, time
from PIL import Image

class UI:
    def __init__(self):
        self.running = True

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
        app = ctk.CTk()
        app.title(title)
        app.attributes('-fullscreen', True)
        app.configure(fg_color="#101010") 

        images = self.load_images(image_path, (340, 340))
 
        bottom_frame = ctk.CTkFrame(app, fg_color="#101010")
        bottom_frame.pack(side='bottom', fill='x')

        ptt_button = ctk.CTkButton(
            bottom_frame,
            text="Push To Talk",
            command=lambda: core.toggle_listening(),
            corner_radius=10,
            font=("Arial", 16, "bold"),
            fg_color="red",
            hover_color="dark red"
        )
        ptt_button.pack(ipady=20, pady=10, padx=20, anchor='sw', side='left')
        
        exit_button = ctk.CTkButton(
            bottom_frame,
            text="Exit",
            command=lambda: self.exit_app(),
            corner_radius=10,
            font=("Arial", 16, "bold")
        )

        cheat_button = ctk.CTkButton(
            bottom_frame, 
            text="Cheat Mode", 
            command=lambda: core.toggle_cheat_mode(),
            corner_radius=10,
            font=("Arial", 16, "bold"),
            fg_color="red"
        )
        
        exit_button.pack(pady=10, padx=20, anchor='se', side='bottom')
        cheat_button.pack(pady=0, padx=20, anchor='se', side='bottom')

        status_label = ctk.CTkLabel(
            app,
            text = "Started!",
            font=("Arial", 20, "bold"),
            text_color="white"    
        )

        sublabel_1 = ctk.CTkLabel(
            app,
            text="Text appears here when submitted.",
            font=("Arial", 20, "bold"),
            text_color="#65FF65"
        )

        sublabel_1.pack(pady=0, anchor='s', side='bottom')
        status_label.pack(pady=0, anchor='s', side='bottom')
        
        label = ctk.CTkLabel(app, text="")
        label.place(relx=0.5, anchor="n")
        new_img = random.choice(images)
        label.configure(image=new_img)

        min_interval = 10
        max_interval = 20

        countdown = random.uniform(min_interval, max_interval)
        start_time = time()
        curr_time = time()
        while core.shared_state["running"]:
            try:
                app.update_idletasks()
                app.update()

                core.shared_state["running"] = self.running
                curr_time = time()
                if (curr_time - start_time) > countdown:
                    new_img = random.choice(images)
                    label.configure(image=new_img)
                    start_time = curr_time
                    countdown = random.uniform(min_interval, max_interval)

                if core.model.cheat_mode:
                    cheat_button.configure(fg_color="green")
                else:
                    cheat_button.configure(fg_color="red")

                if core.shared_state["listening"]:
                    status_label.configure(text="Listening...")
                    sublabel_1.configure(text="Text appears here when submitted.")
                    ptt_button.configure(fg_color="green", hover_color="dark green")
                else:
                    ptt_button.configure(fg_color="red", hover_color="dark red")

                if not core.shared_state["listening"] and len(core.transcribed) != 0:
                    status_label.configure(text="Thinking about what you said:")

                    if len(core.response_buffer) == 0:
                        sublabel_1.configure(text=UI.get_history(core.transcribed))
                    else:
                        sublabel_1.configure(text=UI.get_history(core.response_buffer))

                sleep(0.01)
            except KeyboardInterrupt:
                self.running = False
                try:
                    core.shared_state["running"] = False
                except:
                    pass

                break
            except Exception as e:
                print(f"Error: {e}")
                self.running = False
                try:
                    core.shared_state["running"] = False
                except:
                    pass
                
                break
        
        if core.shared_state["running"]:
            core.shared_state["running"] = self.running
        app.destroy()

        core.invoker_thread.join()