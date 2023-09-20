from pymongo import MongoClient
import os

# Insert connection string and db name:
client = MongoClient(os.environ.get('MONGODB','CONNECTION_STRING'))
db = client[os.environ.get('MONGODB','DB_NAME')]

chats_db = db["chats"]
messages_db = db["messages"]