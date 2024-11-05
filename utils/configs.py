import yaml

def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)