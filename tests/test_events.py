from app.events.generator import EventGenerator

def test_event_generator():
    events = EventGenerator().from_tracking([{"id":"worker_01","class_name":"person","confidence":.95,"position":[100,200]}], 1.5)
    assert len(events) == 1
    assert events[0].entity_id == "worker_01"
    assert events[0].action == "detected"
