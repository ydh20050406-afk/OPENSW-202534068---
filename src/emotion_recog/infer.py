import os
import sys
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


def resource_path(relative_path):
    """파일 경로를 제대로 찾기 위한 함수"""
    if hasattr(sys, '_MEIPASS'):
        
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


MODEL_PATH = resource_path("model")

# 모델 로드
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)


def infer_image(image_path: str):
    """이미지를 받아 감정(label, 확률)을 반환"""
    img = Image.open(image_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    score, idx = torch.max(probs, dim=0)

    label = model.config.id2label[idx.item()]
    return label, float(score)
