from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "dima806/facial_emotions_image_detection"

print("다운중")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
processor.save_pretrained("./model")

model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.save_pretrained("./model")

print("다운완료")
