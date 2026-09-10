class Detector:
    """Model-agnostic detector interface. ROCm-compatible backend later."""
    def __init__(self, model=None):
        self.model = model
    def detect(self, frame):
        raise NotImplementedError
