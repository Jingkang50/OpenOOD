class BasePreprocessor():
    def __init__(self, config):
        self.config = config

    def setup(self):
        return self

    def __call__(self, image):
        return image
