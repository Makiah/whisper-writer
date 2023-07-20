import pyautogui

class ComputerController:
    def __init__(self):
        self.scroll_mode_instructions = {
            "down": lambda: self.scroll(-20),
            "up": lambda: self.scroll(20)
        }
        self.known_instructions = {
            "enter scroll mode": lambda: self.add_context(self.scroll_mode_instructions),
            "exit scroll mode": lambda: self.remove_context(self.scroll_mode_instructions),
            "scroll down": lambda: self.scroll(-20),
            "scroll up": lambda: self.scroll(20)
        }
        self.contexts = [self.known_instructions]

    def scroll(self, amount):
        pyautogui.scroll(amount)

    def add_context(self, context):
        self.contexts.append(context)

    def remove_context(self, context):
        self.contexts.remove(context)
    
    def process(self, instruction: str):
        for context in self.contexts:
            if instruction in context:
                context[instruction]()
                break
        print("No command found")
