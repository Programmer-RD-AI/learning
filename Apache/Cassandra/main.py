from cassandra.cluster import Cluster

# Connect to a Cassandra cluster (default localhost)
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('mykeyspace')

# Execute a simple query
rows = session.execute("SELECT id, name FROM users")
for row in rows:
    print(f"User {row.id}: {row.name}")

cluster.shutdown()
