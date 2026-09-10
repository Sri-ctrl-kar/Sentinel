class EventMemory:
    def __init__(self): self.events = []
    def add(self, event): self.events.append(event)
    def all(self): return list(self.events)
