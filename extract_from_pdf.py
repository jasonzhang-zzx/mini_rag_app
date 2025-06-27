from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import nltk
from nltk.tokenize import sent_tokenize
import json
import os
import fitz
from PIL import Image
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path


def extract_text_from_pdf(pdf_path, page_numbers=None, min_line_length=2):
    """extract text from pdf by specified page numbers"""
    paragraphs = []
    buffer = ""
    full_text = ""

    for i, page_layout in enumerate(extract_pages(pdf_path)):
        if page_numbers is None or i in page_numbers:
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + "\n"
    
    lines = full_text.split("\n")
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (" " + text) if not text.endswith("-") else text.strip("-")
        elif buffer:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def split_text(paragraphs, chunk_size=1800, overlap_size=300):
    """split text into chunks"""
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i-1
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap + chunk
        next = i+1
        while next < len(sentences) and len(chunk) + len(sentences[next]) <= chunk_size:
            chunk += ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks

def process_pdfs_to_json(pdf_paths, output_dir, chunk_size=1800, overlap=300, max_chunks_per_file=500):
    """
    核心处理函数
    :param pdf_paths: PDF文件路径列表
    :param output_dir: 输出目录
    :param chunk_size: 块大小
    :param overlap: 重叠大小
    :param max_chunks_per_file: 单个JSON最大块数
    """
    max_data = []
    os.makedirs(output_dir, exist_ok=True)
    num_json = 0
    for pdf_path in pdf_paths:
        # 获取基础信息
        pdf_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_name}...")
        
        # 提取文本和页码
        paragraphs = extract_text_from_pdf(pdf_path)
        # 分割文本块
        chunks = split_text(paragraphs, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"{Path(pdf_name[:10]).stem}_chunk{idx+1:04d}",
                "text": chunk,
                "source": pdf_name
            }
            max_data.append(record)
            if len(max_data) == max_chunks_per_file:
                filename = f"chunks_part_{num_json + 1}.json"
                output_path = os.path.join(output_dir, filename)
                save_chunks(max_data, output_path)
                max_data = []
                num_json += 1
        
        # 保存剩余的块到新的JSON文件中
        if max_data:
            filename = f"chunks_part_{num_json + 1}.json"
            output_path = os.path.join(output_dir, filename)
            save_chunks(max_data, output_path)


def save_chunks(records, output_path):
    """save chunks to a json file"""
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} chunks to {output_path}")

def load_chunks(json_path):
    """load chunks from a json file"""
    with open(json_path, 'r') as f:
        chunks = json.load(f)
    return chunks

def pdf2images(pdf_path, dpi=300):
    """convert pdf to PNG images"""
    output_folder = os.path.splitext(pdf_path)[0]
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    pdf_document = fitz.open(pdf_path)  # open the PDF file 

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]

        pix = page.get_pixmap(dpi=dpi)  # render page to an image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(f"{output_folder}/page_{page_number + 1}.png")
    
    pdf_document.close()

def show_images(dir_path):
    """show images in a directory"""
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".png"):
            img = Image.open(os.path.join(dir_path, file_name))
            plt.imshow(img)  
            plt.axis('off')  
            plt.show()

class MaxResize(object):
    """Resize the image"""
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
            )
        return resized_image
    

