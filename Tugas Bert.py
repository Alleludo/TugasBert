import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# 1. Setup Awal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Dataset dengan Error Handling
def load_dataset_safe(name, sample_size=1000):
    try:
        dataset = load_dataset(name)
    except:
        try:
            dataset = {
                "train": load_dataset(name, split="train"),
                "test": load_dataset(name, split="test"),
                "validation": load_dataset(name, split="validation") if name == "snli" else None
            }
        except:
            raise ValueError(f"Failed to load {name}")
    
    # Batasi jumlah sampel
    limited_dataset = {}
    for split in dataset:
        if dataset[split] is not None:
            limited_dataset[split] = dataset[split].select(range(min(sample_size, len(dataset[split]))))
        else:
            limited_dataset[split] = None
    
    return limited_dataset

imdb = load_dataset_safe("imdb", 1000)
snli = load_dataset_safe("snli", 1000)

# 3. Tokenizer dan Model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device)

# 4. Strategi Pengambilan Embedding
def get_bert_embeddings(texts, strategy):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    hidden_states = outputs.hidden_states  # Tuple (13 layers)
    
    if strategy == "first_layer":
        return hidden_states[1]  # Layer 1
    elif strategy == "last_layer":
        return hidden_states[12]  # Layer 12
    elif strategy == "sum_all":
        return torch.sum(torch.stack(hidden_states[1:13]), dim=0)  # Sum layer 1-12
    elif strategy == "second_last":
        return hidden_states[11]  # Layer 11
    elif strategy == "sum_last4":
        return torch.sum(torch.stack(hidden_states[9:13]), dim=0)  # Sum layer 9-12
    elif strategy == "concat_last4":
        return torch.cat(hidden_states[9:13], dim=-1)  # Concat layer 9-12 (3072 dim)
    else:
        raise ValueError("Strategi tidak valid")

# 5. Dataset Class
class NLIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "text": f"{item['premise']} [SEP] {item['hypothesis']}",
            "label": item['label']
        }

# 6. Model Klasifikasi
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x[:, 0, :])  # Ambil embedding [CLS]

# 7. Fungsi Training dan Evaluasi
def train_and_evaluate(task_name, train_data, test_data, strategy, num_classes=2, epochs=1, batch_size=32):
    # Siapkan DataLoader
    if task_name == "nli":
        train_dataset = NLIDataset(train_data)
        test_dataset = NLIDataset(test_data)
    else:
        train_dataset = train_data
        test_dataset = test_data
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Inisialisasi Model
    input_dim = 768 if strategy != "concat_last4" else 3072
    model = Classifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            texts = batch["text"]
            labels = torch.tensor(batch["label"]).to(device)
            
            embeddings = get_bert_embeddings(texts, strategy)
            outputs = model(embeddings)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Evaluasi
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            texts = batch["text"]
            labels = batch["label"]
            
            embeddings = get_bert_embeddings(texts, strategy)
            outputs = model(embeddings)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return accuracy_score(all_labels, all_preds)

# 8. Eksperimen Utama
strategies = [
    "first_layer",
    "last_layer",
    "sum_all",
    "second_last",
    "sum_last4",
    "concat_last4"
]

results = {}

# Sentiment Analysis (IMDb)
print("\n" + "="*50)
print("Running Sentiment Analysis Experiments")
print("="*50)
for strategy in strategies:
    acc = train_and_evaluate(
        "sentiment", 
        imdb["train"], 
        imdb["test"], 
        strategy,
        num_classes=2
    )
    results[f"imdb_{strategy}"] = acc
    print(f"IMDb {strategy}: Accuracy = {acc:.4f}")

# Natural Language Inference (SNLI)
print("\n" + "="*50)
print("Running NLI Experiments")
print("="*50)
for strategy in strategies:
    acc = train_and_evaluate(
        "nli",
        snli["train"],
        snli["test"],
        strategy,
        num_classes=3
    )
    results[f"snli_{strategy}"] = acc
    print(f"SNLI {strategy}: Accuracy = {acc:.4f}")

# 9. Visualisasi Hasil
import matplotlib.pyplot as plt

strategies_names = [s.replace("_", " ").title() for s in strategies]
imdb_scores = [results[f"imdb_{s}"] for s in strategies]
snli_scores = [results[f"snli_{s}"] for s in strategies]

plt.figure(figsize=(12, 6))
x = np.arange(len(strategies_names))
width = 0.35

plt.bar(x - width/2, imdb_scores, width, label='IMDb')
plt.bar(x + width/2, snli_scores, width, label='SNLI')

plt.xlabel('Embedding Strategy')
plt.ylabel('Accuracy')
plt.title('Performance Comparison Across Strategies')
plt.xticks(x, strategies_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
