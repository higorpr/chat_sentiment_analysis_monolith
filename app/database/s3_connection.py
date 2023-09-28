import configparser

config = configparser.ConfigParser()
config.read("config.ini")

s3_bucket = config["AWS"]["S3_BUCKET_NAME"]
