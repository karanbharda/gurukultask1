from rag import *
from dotenv import load_dotenv
import uvicorn
import requests
import os
from datetime import datetime
from subject_data import subjects_data
from lectures_data import lectures_data
from test_data import test_data
from db import pdf_collection , image_collection, user_collection
from datetime import datetime, timezone
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

#Chatbot
groq_api_key = os.environ.get('GROQ_API_KEY')

# Temporary in-memory storage
user_queries: List[dict] = []
llm_responses: List[dict] = []

# Pydantic model for request validation
class ChatMessage(BaseModel):
    message: str
    timestamp: str = None
    type: str = "chat_message"


# Route 1: Receive query from frontend
@app.post("/chatpost")
async def receive_query(chat: ChatMessage):
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')    
    query_record = {
        "message": chat.message,
        "timestamp": timestamp,
        "type": "chat_message"
    }
    try:
        chat_collection = user_collection.insert_one(query_record)
        query_record["_id"] = str(chat_collection.inserted_id)  # Add the MongoDB ID to the record
        print(f"Received message: {chat.message}")
        return {"status": "Query received", "data": query_record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store query:{str(e)}")


import requests

def call_groq_llama3(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",  # As per Groq model name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 1.0,
        "stop": None
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

        response.raise_for_status()  # Raise error for bad status codes

        result = response.json()

        return result['choices'][0]['message']['content'].strip()

    except requests.exceptions.RequestException as e:
        print(f"Error calling Groq Llama3 API: {e}")
        return "Failed to fetch response from Groq Llama3 model."


# Route 2: Send LLM response back
@app.get("/chatbot")
async def send_response():
    try:
        # Fetch the latest query from MongoDB
        latest_query = user_collection.find_one({"type": "chat_message", "respone":None}, sort=[("timestamp", -1)])
        if not latest_query:
            return {"error": "No queries yet"}

        query_message = latest_query["message"]
        print(f"Processing query: {query_message}")

        llm_reply = call_groq_llama3(query_message)

        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        response_data = {
            "message": llm_reply,
            "timestamp": timestamp,
            "type": "chat_response",
            "query_id": str(latest_query["_id"])  # Link response to query
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
        raise HTTPException(status_code=500, detail=f"Failed to process response:{str(e)}")

@app.post("/process-pdf", response_model=PDFResponse)
async def process_pdf(file: UploadFile = File(...)):
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
        groq_api_key = os.getenv("GROQ_API_KEY")
        agent = build_qa_agent([structured_data["body"]], groq_api_key=groq_api_key)
        result = agent.invoke({"query": query})
        answer = result["result"]

        audio_file = text_to_speech(answer, file_prefix="output_pdf")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        # Store to MongoDB
        pdf_doc = {
            "filename": file.filename,
            "title": structured_data["title"],
            "sections": [{"heading": s["heading"], "content": s["content"]} for s in structured_data["sections"]],
            "query": query,
            "answer": answer,
            "audio_file": audio_url,
            "timestamp": datetime.now(timezone.utc)
        }
        pdf_collection.insert_one(pdf_doc)

        global pdf_response
        pdf_response = PDFResponse(
            title=structured_data["title"],
            sections=[Section(heading=s["heading"], content=s["content"]) for s in structured_data["sections"]],
            query=query,
            answer=answer,
            audio_file=audio_url
        )
        return pdf_response

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


@app.post("/process-img", response_model=ImageResponse)
async def process_image(file: UploadFile = File(...)):
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
            groq_api_key = os.getenv("GROQ_API_KEY")
            agent = build_qa_agent([ocr_text], groq_api_key=groq_api_key)
            result = agent.invoke({"query": query})
            answer = result["result"]

        audio_file = text_to_speech(answer, file_prefix="output_image")
        audio_url = f"/static/{os.path.basename(audio_file)}" if audio_file else "No audio generated"

        # Store to MongoDB
        image_doc = {
            "filename": file.filename,
            "ocr_text": ocr_text,
            "query": query,
            "answer": answer,
            "audio_file": audio_url,
            "timestamp": datetime.now(timezone.utc)
        }
        image_collection.insert_one(image_doc)

        global image_response
        image_response = ImageResponse(
            ocr_text=ocr_text,
            query=query,
            answer=answer,
            audio_file=audio_url
        )
        return image_response

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

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
        uvicorn.run(app, host="192.168.0.111", port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

