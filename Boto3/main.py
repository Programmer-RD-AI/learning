import boto3

session = boto3.Session(
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    region_name="your-region",
)
s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
s3 = boto3.client("s3")
response = s3.list_buckets()
for bucket in response["Buckets"]:
    print(bucket["Name"])
s3.create_bucket(Bucket="my-new-bucket")
s3.upload_file("local_file.txt", "my-bucket", "s3_file.txt")
s3.download_file("my-bucket", "s3_file.txt", "local_file.txt")
ec2 = boto3.client("ec2")
response = ec2.describe_instances()
for reservation in response["Reservations"]:
    for instance in reservation["Instances"]:
        print(instance["InstanceId"])
ec2.start_instances(InstanceIds=["i-1234567890abcdef"])
ec2.stop_instances(InstanceIds=["i-1234567890abcdef"])
dynamodb = boto3.resource("dynamodb")
table = dynamodb.create_table(
    TableName="my-table",
    KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
    ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
)
table.put_item(Item={"id": "123", "name": "John Doe"})
response = table.get_item(Key={"id": "123"})
print(response["Item"])
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

try:
    s3.upload_file("local_file.txt", "my-bucket", "s3_file.txt")
except NoCredentialsError:
    print("Credentials not available")
except PartialCredentialsError:
    print("Incomplete credentials provided")
