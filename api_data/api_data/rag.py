from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import logging
from fastapi.responses import FileResponse
import os
import fitz  # PyMuPDF
import easyocr
from paddleocr import PaddleOCR, draw_ocr
import re
import time
import socket
import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import pytesseract
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from gtts import gTTS
from typing import List, Dict
import shutil
import logging
from langchain.llms.base import LLM
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temporary directory for files
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Pydantic models for response structure
class Section(BaseModel):
    heading: str
    content: str

class PDFResponse(BaseModel):
    title: str
    sections: List[Section]
    query: str
    answer: str
    audio_file: str

class ImageResponse(BaseModel):
    ocr_text: str
    query: str
    answer: str
    audio_file: str

pdf_response: PDFResponse | None = None
image_response: ImageResponse| None = None

class SimpleGroqLLM(LLM):
    groq_api_key: str
    model: str = "llama3-8b-8192"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        
        try:
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format from Groq API: {result}")
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise RuntimeError("Failed to generate response from Groq API.")
    
    @property
    def _llm_type(self) -> str:
        return "groq-llm"
    
def parse_pdf(file_path: str) -> Dict:
    try:
        doc = fitz.open(file_path)
        raw_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            raw_text += text + "\n"
        doc.close()
        lines = raw_text.strip().split('\n')
        title = next((line.strip() for line in lines if line.strip()), "")
        section_pattern = re.compile(r'^(?:\d+\.?)+\s+.+', re.MULTILINE)
        matches = list(section_pattern.finditer(raw_text))
        sections = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            section_title = match.group().strip()
            section_body = raw_text[start:end].strip()
            sections.append({"heading": section_title, "content": section_body})
        return {
            "title": title,
            "body": raw_text.strip(),
            "sections": sections
        }
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return {"title": "", "body": "", "sections": []}

def extract_text_easyocr(image_path: str) -> str:
    reader = easyocr.Reader(['en' , 'hi'], gpu=False)
    result = reader.readtext(image_path, detail=0)
    print("OCR result list:", result)
    return " ".join(result)

#def extract_text_tesseract(image_path: str) -> str:
#    try:
#        # Use 'with' to automatically close the file
#        with Image.open(image_path) as img:
#            # Use Tesseract to extract text
#           extracted_text = pytesseract.image_to_string(img, lang='eng+hin')
#
#        print("OCR result:", extracted_text)
#        return extracted_text.strip()
#    
#    except Exception as e:
#        print(f"Error during OCR: {e}")
#        raise e

def build_qa_agent(texts: List[str], groq_api_key: str) -> RetrievalQA:
    llm = SimpleGroqLLM(groq_api_key=groq_api_key, model="llama3-8b-8192")
    documents = [Document(page_content=t) for t in texts if t.strip()]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )
    return qa

def text_to_speech(text: str, file_prefix: str = "output") -> str:
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(TEMP_DIR, f"{file_prefix}_{timestamp}.mp3")
        logger.info(f"Generating audio with Google TTS to {output_file}")
        tts = gTTS(text=text, lang="en")
        tts.save(output_file)
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logger.error(f"Audio file {output_file} was not created or is empty")
            return ""
        return output_file
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return ""

