from rag import *
from dotenv import load_dotenv
import uvicorn
import requests
import os
from datetime import datetime
from subject_data import subjects_data
from lectures_data import lectures_data
from test_data import test_data
from db import user_collection, pdf_collection, image_collection
from datetime import datetime, timezone
from fastapi import HTTPException, FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shutil
import time
import logging
from typing import Optional, List

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Subject API Models and Routes ====
class Subject(BaseModel):
    id: int
    name: str
    code: str

@app.get("/subjects_dummy")
def get_subjects():
    return JSONResponse(content=subjects_data)

# ==== Lectures API Models and Routes ====
class Lecture(BaseModel):
    id: int
    topic: str
    subject_id: int

@app.get("/lecture_dummy")
def get_lectures():
    return JSONResponse(content=lectures_data)

# ==== Test API Models and Routes ====
class Test(BaseModel):
    id: int
    name: str
    subject_id: int
    date: Optional[str] = None

@app.get("/test_dummy")
def get_test():
    return JSONResponse(content=test_data)

# ==== Chatbot Models and Routes ====
groq_api_key = os.environ.get('GROQ_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
llama_api_key = os.environ.get('LLAMA_API_KEY')
uniguru_api_key = os.environ.get('UNIGURU_API_KEY')

# Temporary in-memory storage
user_queries: List[dict] = []
llm_responses: List[dict] = []

# Pydantic model for chat request
class ChatMessage(BaseModel):
    message: str
    llm: str = Field(..., pattern = "^(grok|llama|chatgpt|uniguru)$")
    timestamp: str = None
    type: str = "chat_message"

# Pydantic models for PDF and Image (assumed, based on response_model)
class Section(BaseModel):
    heading: str
    content: str

class PDFResponse(BaseModel):
    title: str
    sections: List[Section]
    query: str
    answer: str
    audio_file: str
    llm: str  # Added to store selected LLM

class ImageResponse(BaseModel):
    ocr_text: str
    query: str
    answer: str
    audio_file: str
    llm: str  # Added to store selected LLM

# Global variables for storing latest responses
pdf_response = None
image_response = None

def call_llm(prompt: str, llm: str) -> str:
    """
    Call the specified LLM API with the given prompt.
    """
    try:
        if llm == "grok":
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 1.0
            }
            if not prompt.strip():
                raise ValueError("Prompt must not be empty")

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        elif llm == "llama":
            headers = {
                "Authorization": f"Bearer {groq_api_key}",  # Using the same Groq API key
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-70b-8192",  # Groq's LLaMA 3 model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 1.0
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        elif llm == "chatgpt":
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 1.0
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        elif llm == "uniguru":
            headers = {
                "Authorization": f"Bearer {uniguru_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.7
            }
            response = requests.post(
                "https://api.uniguru.com/v1/completions",  # Replace if necessary
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result['text'].strip()  # Adjust based on actual response

        else:
            raise ValueError(f"Unsupported LLM: {llm}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling {llm} API: {e}")
        return f"Failed to fetch response from {llm} model."
    except Exception as e:
        logger.error(f"Unexpected error with {llm}: {e}")
        return "An unexpected error occurred."
    
# Chatbot Routes
@app.post("/chatpost")
async def receive_query(chat: ChatMessage):
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    query_record = {
        "message": chat.message,
        "llm": chat.llm,
        "timestamp": timestamp,
        "type": "chat_message"
    }
    try:
        chat_collection = user_collection.insert_one(query_record)
        query_record["_id"] = str(chat_collection.inserted_id)
        logger.info(f"Received message: {chat.message} for LLM: {chat.llm}")
        return {"status": "Query received", "data": query_record}
    except Exception as e:
        logger.error(f"Failed to store query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store query: {str(e)}")

@app.get("/chatbot")
async def send_response():
    try:
        latest_query = user_collection.find_one(
            {"type": "chat_message", "response": None},
            sort=[("timestamp", -1)]
        )
        if not latest_query:
            return {"error": "No queries yet"}

        query_message = latest_query["message"]
        selected_llm = latest_query["llm"]
        logger.info(f"Processing query: {query_message} with LLM: {selected_llm}")

        llm_reply = call_llm(query_message, selected_llm)

        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        response_data = {
            "message": llm_reply,
            "timestamp": timestamp,
            "type": "chat_response",
            "query_id": str(latest_query["_id"]),
            "llm": selected_llm
        }

        user_collection.update_one(
            {"_id": latest_query["_id"]},
            {"$set": {"response": response_data}}
        )

        return {
            "_id": str(latest_query["_id"]),
            "query": query_message,
            "response": response_data
        }

    except Exception as e:
        logger.error(f"Failed to process response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process response: {str(e)}")

# PDF Processing Route
@app.post("/process-pdf", response_model=PDFResponse)
async def process_pdf(file: UploadFile = File(...), llm: str = Form(..., pattern="^(grok|llama|chatgpt|uniguru)$")):
    temp_pdf_path = ""
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_pdf_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
        with open(temp_pdf_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        structured_data = parse_pdf(temp_pdf_path)
        if not structured_data["body"]:
            raise HTTPException(status_code=400, detail="Failed to parse PDF content")

        query = "give me detail summary of this pdf"
        # Use call_llm instead of build_qa_agent for consistency
        answer = call_llm(f"Summarize the following content: {structured_data['body']}", llm)

        audio_file = text_to_speech(answer, file_prefix="output_pdf")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        pdf_doc = {
            "filename": file.filename,
            "title": structured_data["title"],
            "sections": [{"heading": s["heading"], "content": s["content"]} for s in structured_data["sections"]],
            "query": query,
            "answer": answer,
            "audio_file": audio_url,
            "llm": llm,  # Store selected LLM
            "timestamp": datetime.now(timezone.utc)
        }
        pdf_collection.insert_one(pdf_doc)

        global pdf_response
        pdf_response = PDFResponse(
            title=structured_data["title"],
            sections=[Section(heading=s["heading"], content=s["content"]) for s in structured_data["sections"]],
            query=query,
            answer=answer,
            audio_file=audio_url,
            llm=llm
        )
        return pdf_response

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# Image Processing Route
@app.post("/process-img", response_model=ImageResponse)
async def process_image(file: UploadFile = File(...), llm: str = Form(..., pattern="^(grok|llama|chatgpt|uniguru)$")):
    temp_image_path = ""
    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are allowed")

        temp_image_path = os.path.join(
            TEMP_DIR,
            f"temp_image_{time.strftime('%Y%m%d_%H%M%S')}{os.path.splitext(file.filename)[1]}"
        )

        with open(temp_image_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        ocr_text = extract_text_easyocr(temp_image_path).strip()
        logger.info(f"OCR raw output: {repr(ocr_text)}")

        if not ocr_text:
            ocr_text = "No readable text found in the image."
            answer = ocr_text
            query = "N/A"
        else:
            query = "give me detail summary of this image"
            answer = call_llm(f"Summarize the following text extracted from an image: {ocr_text}", llm)

        audio_file = text_to_speech(answer, file_prefix="output_image")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        image_doc = {
            "filename": file.filename,
            "ocr_text": ocr_text,
            "query": query,
            "answer": answer,
            "audio_file": audio_url,
            "llm": llm,  # Store selected LLM
            "timestamp": datetime.now(timezone.utc)
        }
        image_collection.insert_one(image_doc)

        global image_response
        image_response = ImageResponse(
            ocr_text=ocr_text,
            query=query,
            answer=answer,
            audio_file=audio_url,
            llm=llm
        )
        return image_response

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# Summarize Routes
@app.get("/summarize-pdf", response_model=PDFResponse)
async def summarize_pdf():
    if pdf_response is None:
        raise HTTPException(status_code=404, detail="No PDF has been processed yet.")
    return pdf_response

@app.get("/summarize-img", response_model=ImageResponse)
async def summarize_image():
    if image_response is None:
        raise HTTPException(status_code=404, detail="No image has been processed yet.")
    return image_response

# Audio Streaming and Download Routes
@app.get("/api/stream/{filename}")
async def stream_audio(filename: str):
    audio_path = os.path.join(TEMP_DIR, filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.get("/api/audio/{filename}")
async def download_audio(filename: str):
    audio_path = os.path.join(TEMP_DIR, filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename
    )

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="192.168.0.73", port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise