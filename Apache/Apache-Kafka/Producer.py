from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# Send a sample message to topic 'test-topic'
producer.send("test-topic", {"event": "sample", "value": 123})
producer.flush()
