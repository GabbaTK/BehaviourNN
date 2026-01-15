import json
import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Config
PORT = 3725
CHUNK_SIZE = 512
FORMAT = "utf-8"
TARGET_LOSS = 0.33
MAX_EPOCH = 100000
SMOOTHING = 0.1
DROPOUT_PCT = 0.05

words = [
    'river', 'whistle', 'bucket', 'bell',
    'bottle', 'shirt', 'candle', 'path', 
    'cupboard', 'bridge', 'harbor', 'ball', 
    'dock', 'lantern', 'trail', 'curtain', 
    'torch', 'pen', 'book', 'knife', 'plateau', 
    'blanket', 'apple', 'hill', 'plant', 
    'rain', 'riverbank', 'flower', 'cave', 
    'shadow', 'sky', 'lamp', 'forest', 
    'painting', 'mountain', 'hat', 'alley', 
    'keyboard', 'cup', 'clocktower', 'water', 
    'attic', 'canyon', 'wheel', 'paper', 
    'ring', 'forestpath', 'basement', 'tree', 
    'rope', 'snow', 'screen', 'ladder', 
    'cable', 'key', 'anchor', 
    'leaf', 'tower', 'canvas', 'island', 
    'cliff', 'cloud', 'star', 'glade', 
    'crystal', 'fountain', 'plane', 'tunnel', 
    'brush', 'mirror', 'picture', 'roadway', 
    'statue', 'clouds', 'shed', 'house', 
    'glass', 'chair', 'pond', 'car', 'clock', 
    'fire', 'bench', 'sculpture', 'map', 
    'rock', 'mouse', 'wall', 'ship', 'door', 
    'moon', 'stone', 'rug', 'basket', 'button', 
    'coin', 'street', 'sun', 'sand', 'valley', 
    'lake', 'letter', 'window', 'fog', 
    'wind', 'doorway', 'train', 'road', 'meadow',
    'garage', 'boat', 'plate', 'pillow', 'garden', 
    'field'
    ]
words_ids = {w: i for i, w in enumerate(words)}
VOCAB_SIZE = len(words)

# Stores
models = {}               # guid -> model
training_status = {}      # guid -> status dict


class ChoiceDataset(Dataset):
    def __init__(self, choices):
        self.data = []

        for entry in choices:
            L = words_ids[entry["L"]]
            R = words_ids[entry["R"]]
            label = SMOOTHING if entry["C"] == entry["L"] else 1 - SMOOTHING
            self.data.append((L, R, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        L, R, label = self.data[idx]

        return (
            torch.tensor(L, dtype=torch.long),
            torch.tensor(R, dtype=torch.long),
            torch.tensor([label], dtype=torch.float32)
            )

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PCT),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, L, R):
        L_emb = self.embedding(L)
        R_emb = self.embedding(R)
        x = torch.cat([L_emb, R_emb], dim=1)

        return self.net(x)

def trainModelAsync(guid, choices):
    try:
        training_status[guid] = {
            "status": "training",
            "epoch": 0,
            "loss": None
        }

        dataset = ChoiceDataset(choices)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = Model(VOCAB_SIZE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lossFunc = nn.BCEWithLogitsLoss()

        for epoch in range(1, MAX_EPOCH + 1):
            correct = 0
            total = 0
            totalLoss = 0.0

            for L, R, y in loader:
                logits = model(L, R)
                loss = lossFunc(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                totalLoss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

            #acc = correct / total
            avg_loss = totalLoss / len(loader)

            training_status[guid].update({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                #"accuracy": round(acc, 4)
            })

            #if acc >= TARGET_ACC:
            if avg_loss <= TARGET_LOSS:
                break

        models[guid] = model.eval()
        torch.save(model.state_dict(), f"model.{guid}.pt")

        training_status[guid]["status"] = "complete"

    except Exception as e:
        training_status[guid] = {
            "status": "error",
            "message": str(e)
        }

def loadSavedModels():
    files = os.listdir()

    for file in files:
        if not os.path.isfile(file): continue

        if file.startswith("model."):
            model = Model(VOCAB_SIZE)
            model.load_state_dict(torch.load(file))
            model.eval()

            guid = file[6:-3] # Remove model. and .pt

            models[guid] = model
            training_status[guid] = {
                "status": "complete"
            }

def handleClient(conn):
    try:
        rawLen = conn.recv(4)
        msgLen = int.from_bytes(rawLen, "big")

        allData = b""
        while len(allData) < msgLen:
            allData += conn.recv(msgLen - len(allData))

        allData = allData.decode(FORMAT)
        print(allData[:20], end=" ... ")
        print(allData[-20:])

        req = json.loads(allData)

        guid = req.get("guid")
        if not guid:
            raise ValueError("Missing guid")

        # Status request
        if req.get("status") is True:
            response = training_status.get(guid, {
                "status": "not_started"
            })

        # Train request
        elif "choices" in req:
            threading.Thread(
                target=trainModelAsync,
                args=(guid, req["choices"]),
                daemon=True
            ).start()

            response = {
                "status": "training_started",
                "guid": guid
            }

        # Inference request
        elif req.get("inference") is True:
            if guid not in models:
                response = {
                    "status": "error",
                    "message": "Model not ready"
                }

            elif training_status[guid]["status"] != "complete":
                response = {
                    "status": "error",
                    "message": "Model not ready"
                }

            else:
                L = words_ids[req["L"]]
                R = words_ids[req["R"]]

                model = models[guid]
                with torch.no_grad():
                    logit = model(
                        torch.tensor([L]),
                        torch.tensor([R])
                    )
                    p_left = torch.sigmoid(logit).item()
                    p_right = 1.0 - p_left

                response = {
                    "P_left": p_left,
                    "P_right": p_right
                }

                print(f"GUID: {guid}, P_left: {round(p_left, 4)} | P_right: {round(p_right, 4)}")


    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }

    conn.send(json.dumps(response).encode(FORMAT))
    conn.close()

def startServer(host="0.0.0.0", port=PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)

    print("Loading models")
    loadSavedModels()

    print(f"Listening on port {port}")

    while True:
        conn, _ = sock.accept()
        threading.Thread(
            target=handleClient,
            args=(conn,),
            daemon=True
        ).start()

if __name__ == "__main__":
    startServer()
