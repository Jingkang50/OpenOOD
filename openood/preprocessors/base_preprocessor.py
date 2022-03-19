class BasePreprocessor():
    def __init__(self, config=None):
        self.config = config

    # append transforms that will apply after the preprocessor
    def concat_transform(self, post_preprocessor_transform=None):
        self.post_preprocessor_transform = post_preprocessor_transform
        return self

    def __call__(self, img):
        # apply transforms to all returned images, in base case only one return
        if self.post_preprocessor_transform is not None:
            img = self.post_preprocessor_transform(img)
        return img