class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)