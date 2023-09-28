from pymongo import MongoClient
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Insert connection string and db name:
client = MongoClient(config["MONGODB"]["DEV_CONNECTION_STRING"])
db = client[config["MONGODB"]["DEV_DB_NAME"]]

chats_db = db["chats"]
messages_db = db["messages"]
