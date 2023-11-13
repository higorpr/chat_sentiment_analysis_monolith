from app.database.connection import chats_db, messages_db
from bson.objectid import ObjectId


class ReportRepository:
    def __init__(self):
        self.chat_collection = chats_db
        self.message_collection = messages_db

    def get_chat_id(self, account_id: str, wa_chat_id: str):
        query = {"account": ObjectId(account_id), "wa_chat_id": wa_chat_id}
        output = self.chat_collection.find(query)
        # This conversion is here to be able to know if the query returned any results
        output = list(output)
        return output

    def get_chat_messages(self, chat_id: str):
        query = {"chat": ObjectId(chat_id), "type": "chat", "text": {"$exists": "true"}}
        return self.message_collection.find(query).sort("send_date", 1)
