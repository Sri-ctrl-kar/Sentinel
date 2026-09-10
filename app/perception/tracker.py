class Tracker:
    """Model-agnostic tracking interface."""
    def update(self, detections, timestamp):
        raise NotImplementedError
