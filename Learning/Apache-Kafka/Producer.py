# Sends events too the Kafka Broker
# Responsible For:
#   - Partition Assignment
#   - Batching Events for Improved Throughput
#   - Compression
#   - Retries
#   - Response callbacks
#   - Transaction Handling

# Configs: acks
#        : 0 -> producer requests are sent but no guarantee that it made it to the broker, fast but unsafe
#        : 1 -> Makes sure that the lead broker recieves the data and is stored but no guarantee if the replicas has been done.
#        : -1 -> Guarantee that the lead broker and the replicas are updated

# Configs: Batch Size
# Target size for a request call, this can reduce the number of network calls by combing several events, but it will increase latency

# Configs: Linger.MS
# Number of milliseconds to wait before sending a batch, this can increase the number of events in a batch

# Configs: Compression
# Compression Type: gzip, snappy, lz4, zstd

# Configs: Retries
# Number of retries before giving up on sending a record

# Configs: Delivery Timeout
# The maximum amount of time the producer will wait for the broker to acknowledge receipt of a record

# Config: Transactional ID
# A unqiue id for the producer that enables transaction recovery across multiple sessions

# Config: Enable Idempotence
# When set true, the producer adds a unique sequence number to messages

# key = KCSTXDP3J6OHP2IQ
# secret key = SK4W2OWmiNs1Tn7zN8LoSwrP6QFDGBiUmml3YcZl2V1HANcqeySPB5UARa7AqEeC

from confluent_kafka import Producer, Consumer


def read_config():
    # reads the client configuration from client.properties
    # and returns it as a key-value map
    config = {}
    with open("client.properties") as fh:
        for line in fh:
            line = line.strip()
            if len(line) != 0 and line[0] != "#":
                parameter, value = line.strip().split("=", 1)
                config[parameter] = value.strip()
    return config


def produce(topic, config):
    # creates a new producer instance
    producer = Producer(config)

    # produces a sample message
    key = "key"
    value = "value"
    producer.produce(topic, key=key, value=value)
    print(f"Produced message to topic {topic}: key = {key:12} value = {value:12}")

    # send any outstanding or buffered messages to the Kafka broker
    producer.flush()


def consume(topic, config):
    # sets the consumer group ID and offset
    config["group.id"] = "python-group-1"
    config["auto.offset.reset"] = "earliest"

    # creates a new consumer instance
    consumer = Consumer(config)

    # subscribes to the specified topic
    consumer.subscribe([topic])

    try:
        while True:
            # consumer polls the topic and prints any incoming messages
            msg = consumer.poll(1.0)
            if msg is not None and msg.error() is None:
                key = msg.key().decode("utf-8")
                value = msg.value().decode("utf-8")
                print(
                    f"Consumed message from topic {topic}: key = {key:12} value = {value:12}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        # closes the consumer connection
        consumer.close()


def main():
    config = read_config()
    topic = "topic_0"

    produce(topic, config)
    # consume(topic, config)


main()
