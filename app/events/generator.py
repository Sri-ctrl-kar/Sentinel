from .schema import Event

class EventGenerator:
    def from_tracking(self, tracked_objects, timestamp):
        return [Event(timestamp, str(obj["id"]), "detected", {
            "class_name": obj.get("class_name"),
            "confidence": obj.get("confidence")
        }, obj.get("position")) for obj in tracked_objects]
