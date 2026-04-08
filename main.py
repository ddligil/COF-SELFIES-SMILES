import pandas as pd
import selfies as sf
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import numpy as np
import random

# 1. VERİ SETİ VE SÖZLÜK (GENİŞLETİLMİŞ)
print("1. Aşama: Bilimsel veri kütüphanesi hazırlanıyor...")

# Gerçek COF bağlayıcıları ve onların oluşturduğu geometriler
cof_data = [
    {"ID": "HEX_1", "SMILES": "C1=CC(=CC=C1B(O)O)B(O)O", "topo": "hexagonal"},
    {"ID": "HEX_2", "SMILES": "C1=CC=C(C=C1)B2OC(OB(O2)C3=CC=C(C=C3)B)O", "topo": "hexagonal"},
    {"ID": "HEX_3", "SMILES": "c1(N)cc(N)cc(N)c1", "topo": "hexagonal"},
    {"ID": "SQR_1", "SMILES": "B1(OB(OB(O1)C2=CC=C(C=C2)B)O)C3=CC=C(C=C3)B", "topo": "square"},
    {"ID": "SQR_2", "SMILES": "c1(C#Cc2ccc(N)cc2)cc(C#Cc3ccc(N)cc3)cc(C#Cc4ccc(N)cc4)c1", "topo": "square"},
    {"ID": "SQR_3", "SMILES": "C1=CC(=CC=C1C2=CC=C(C=C2)N)N", "topo": "square"}
]

# Veriyi 100 katına çıkarıyoruz (Data Augmentation simülasyonu)
expanded_data = cof_data * 50 
df = pd.DataFrame(expanded_data)
df['SELFIES'] = df['SMILES'].apply(lambda s: sf.encoder(s))

# Sözlük Oluşturma
all_selfies = df['SELFIES'].tolist()
alphabet = sorted(list(sf.get_alphabet_from_selfies(all_selfies)))
if "[nop]" not in alphabet: alphabet.append("[nop]") 
token_to_id = {t: i for i, t in enumerate(alphabet)}
id_to_token = {i: t for i, t in enumerate(alphabet)}
vocab_size = len(alphabet)

# 2. DATASET: ŞEKİLDEN MOLEKÜLE (TASARIMCI MANTIĞI)
class COFDesignDataset(Dataset):
    def __init__(self, dataframe, token_map, max_len=64):
        self.df = dataframe
        self.token_map = token_map
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15), # Veriyi çeşitlendiriyoruz
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        # RESİM: Molekülün kendi resmi (Modelin 'Gözü')
        mol = Chem.MolFromSmiles(self.df.iloc[idx]['SMILES'])
        img = Draw.MolToImage(mol, size=(300, 300)).convert("RGB")
        img_tensor = self.transform(img)
        
        # HEDEF: SELFIES Kodu
        selfie_str = self.df.iloc[idx]['SELFIES']
        encoded = sf.selfies_to_encoding(
            selfies=selfie_str, vocab_stoi=self.token_map, 
            pad_to_len=self.max_len, enc_type="label"
        )
        return {"image": img_tensor, "label": torch.tensor(encoded, dtype=torch.long)}

# 3. MODEL MİMARİSİ (ENCODER-DECODER)
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        return self.linear(features.view(features.size(0), -1))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, labels):
        embeddings = self.embed(labels)
        # Resim özelliklerini dizinin başına ekle
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

# 4. EĞİTİM DÖNGÜSÜ (GÜÇLENDİRİLMİŞ)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size, hidden_size = 256, 512

encoder = Encoder(embed_size).to(device)
decoder = Decoder(embed_size, hidden_size, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

dataset = COFDesignDataset(df, token_to_id)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"2. Aşama: Eğitim {device} üzerinde başlıyor...")
for epoch in range(50): # 50 Epoch daha stabil sonuç verir
    for batch in dataloader:
        imgs, labels = batch['image'].to(device), batch['label'].to(device)
        
        features = encoder(imgs)
        outputs = decoder(features, labels[:, :-1])
        
        loss = criterion(outputs.reshape(-1, vocab_size), labels.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

print("Eğitim Tamamlandı!")

# 5. TEST: ŞEKİLDEN TAHMİN ALMA
def predict(idx):
    sample = dataset[idx]
    img = sample['image'].unsqueeze(0).to(device)
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(img)
        # Tahmin başlangıcı
        inputs = torch.tensor([[token_to_id["[nop]"]]]).to(device)
        result = []
        for _ in range(64):
            outputs = decoder(features, inputs)
            predicted = outputs[:, -1, :].argmax(1).item()
            char = id_to_token[predicted]
            if char == "[nop]" and len(result) > 0: break
            result.append(char)
            inputs = torch.cat((inputs, torch.tensor([[predicted]]).to(device)), dim=1)
    
    print(f"\nID: {df.iloc[idx]['ID']} ({df.iloc[idx]['topo']})")
    print(f"GERÇEK : {df.iloc[idx]['SELFIES']}")
    print(f"TAHMİN : {''.join(result)}")

import gradio as gr
from torchvision import transforms
from PIL import Image
import torch
import selfies as sf

# --- KRİTİK KONTROL ---
# Eğer 'encoder' veya 'decoder' bulunamazsa eğitim hücresini tekrar çalıştırın.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_from_image(input_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(input_img).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Global değişkenleri (modelleri) fonksiyon içine tanıtıyoruz
    global encoder, decoder, token_to_id, id_to_token
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(img_tensor)
        start_id = token_to_id.get("[C]", 0)
        inputs = torch.tensor([[start_id]]).to(device)
        result_tokens = ["[C]"]
        
        for _ in range(60):
            outputs = decoder(features, inputs)
            predicted = outputs[:, -1, :].argmax(1).item()
            char = id_to_token[predicted]
            if char == "[nop]": break
            result_tokens.append(char)
            new_input = torch.tensor([[predicted]]).to(device)
            inputs = torch.cat((inputs, new_input), dim=1)
            
    selfie_output = "".join(result_tokens)
    try:
        smiles_output = sf.decoder(selfie_output)
    except:
        smiles_output = "SMILES çevirisi yapılamadı."
        
    return f"Önerilen SELFIES: {selfie_output}\n\nTahmini SMILES: {smiles_output}"

demo = gr.Interface(fn=predict_from_image, inputs=gr.Image(), outputs="text", title="COF Tasarım Robotu")
demo.launch()

predict(0) # İlk örneği test et
