class BasePreprocessor():
    def __init__(self, config=None):
        self.config = config

    def __call__(self, img):
        return img
