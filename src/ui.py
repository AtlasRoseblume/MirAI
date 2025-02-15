import customtkinter as ctk
import gc
from time import sleep 
from threading import Thread

class UI:
    def __init__(self, core, title: str = "MirAI"):
        self.running = True

        self.background_thread = Thread(target=UI.run_customtkinter, args=(self, core, title))
        self.background_thread.start()
    
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

    def run_customtkinter(self, core, title):
        # ctk.set_default_color_theme("blue")

        app = ctk.CTk()
        app.title(title)
        app.attributes('-fullscreen', True)
        app.configure(fg_color="#1F1F1F") 
 
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
            text="Cheat Code", 
            command=lambda: core.toggle_cheat_code(),
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


        while self.running:
            try:
                app.update_idletasks()
                app.update()

                if core.cheat_code:
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
