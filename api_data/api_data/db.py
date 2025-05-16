from pymongo import MongoClient
import os

# Use environment variable or hardcode your Mongo URI (use Compass URI)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

db = client["gurukul"]

# Define collections
user_collection = db["chat_collection"]
pdf_collection = db["pdf_collection"]        
image_collection = db["image_collection"]

# New collections for subjects, lectures, and tests
subjects_collection = db["subjects_collection"]
lectures_collection = db["lectures_collection"]
tests_collection = db["tests_collection"]
