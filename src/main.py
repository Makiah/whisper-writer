from transcription import record_and_transcribe
import os
import json

def load_config_with_defaults():
    default_config = {
        'use_api': True,
        'api_options': {
            'model': 'whisper-1',
            'language': None,
            'temperature': 0.0,
            'initial_prompt': None
        },
        'local_model_options': {
            'model': 'base',
            'device': None,
            'language': None,
            'temperature': 0.0,
            'initial_prompt': None,
            'condition_on_previous_text': True,
            'verbose': False
        },
        'activation_key': 'ctrl+alt+space',
        'silence_duration': 900,
        'writing_key_press_delay': 0.008,
        'remove_trailing_period': True,
        'add_trailing_space': False,
        'remove_capitalization': False,
        'print_to_terminal': True,
    }

    config_path = os.path.join('config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r') as config_file:
            user_config = json.load(config_file)
            for key, value in user_config.items():
                if key in default_config and value is not None:
                    default_config[key] = value

    return default_config

def run():
    config = load_config_with_defaults()
    method = 'OpenAI\'s API' if config['use_api'] else 'a local model'
    print(f"Listening using {method}")
    record_and_transcribe(config)

if __name__ == "__main__":
    run()
