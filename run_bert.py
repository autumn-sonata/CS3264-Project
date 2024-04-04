from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Load model
model_path = 'bert_model_url_classification\distilBERT Model 3 Epochs 100k Samples'
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(  inputs)
        logits = outputs.logits
    
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence_score = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence_score

label_map = ["benign", "defacement", "phishing", "malware"]

while(True):
  input_text = input("Enter a URL to classify: ")
  predicted_class, confidence = classify_text(input_text)
  print("Predicted class:", label_map[predicted_class], ", Confidence:", f"{confidence:.2f}")