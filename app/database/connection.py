from pymongo import MongoClient
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Insert connection string and db name:
client = MongoClient(config["MONGODB"]["CONNECTION_STRING"])
db = client[config["MONGODB"]["DB_NAME"]]

chats_db = db["chats"]
messages_db = db["messages"]
