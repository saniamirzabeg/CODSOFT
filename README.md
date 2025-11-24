CODSOFT Task 1
Chatbot with Rule Based Responses

print("Chatbot: Hello! I am a simple rule-based chatbot.")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").lower()

    if "hello" in user_input or "hi" in user_input:
        print("Chatbot: Hello! How can I help you today?")
    
    elif "your name" in user_input:
        print("Chatbot: I am a rule-based chatbot created using Python.")
    
    elif "how are you" in user_input:
        print("Chatbot: I'm just a program, but I'm working perfectly!")
    
    elif "weather" in user_input:
        print("Chatbot: I cannot check weather, but I hope it's nice where you are!")
    
    elif "bye" in user_input:
        print("Chatbot: Goodbye! Have a great day!")
        break

    else:
        print("Chatbot: Sorry, I don't understand that. Try asking something else.")


CODSOFT TASK 2
TIC TAC TOE AI

import math
board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(" ", board[3*i], "|", board[3*i+1], "|", board[3*i+2])
        if i < 2:
            print("---|---|---")
    print()
def check_winner(brd, player):
    win_conditions = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for combo in win_conditions:
        if brd[combo[0]] == brd[combo[1]] == brd[combo[2]] == player:
            return True
    return False
def is_draw(brd):
    return " " not in brd
def minimax(brd, depth, is_maximizing):
    if check_winner(brd, "O"):
        return 1
    if check_winner(brd, "X"):
        return -1
    if is_draw(brd):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "O"
                score = minimax(brd, depth + 1, False)
                brd[i] = " "
                best_score = max(best_score, score)
        return best_score

    else:
        best_score = math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = "X"
                score = minimax(brd, depth + 1, True)
                brd[i] = " "
                best_score = min(best_score, score)
        return best_score
def ai_move():
    best_score = -math.inf
    best_move = None

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i

    board[best_move] = "O"
def human_move():
    while True:
        pos = int(input("Enter your move (1-9): ")) - 1
        if 0 <= pos <= 8 and board[pos] == " ":
            board[pos] = "X"
            break
        else:
            print("Invalid move, try again.")


CODSOFT TASK 2
def play():
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, AI is O.")
    print_board()

    while True:
        human_move()
        print_board()

        if check_winner(board, "X"):
            print("You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        print("AI is thinking...")
        ai_move()
        print_board()

            print("It's a draw!")
            break
play()



CODSOFT TASK 3

import os
import math
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import nltk
nltk.download('punkt')

# ---------------------------
# 1) Hyperparams / config
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 3
FFN_DIM = 2048
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-4
MAX_LEN = 30
MIN_WORD_FREQ = 5

# ---------------------------
# 2) Simple Vocabulary class
# ---------------------------
class Vocab:
    def __init__(self, freq_threshold=MIN_WORD_FREQ):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenize(text)
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokenized]

# ---------------------------
# 3) Dataset (expects image files + captions JSON)
# Example annotations format:
# [{"image_path":"path/to/img1.jpg","caption":"A cat sits on a mat."}, ...]
# ---------------------------
class CaptionDataset(Dataset):
    def __init__(self, annotations_file, vocab, transform=None, max_len=MAX_LEN):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img = Image.open(item['image_path']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption = item['caption']
        numericalized = [self.vocab.stoi["<sos>"]] + \
                        self.vocab.numericalize(caption)[:self.max_len-2] + \
                        [self.vocab.stoi["<eos>"]]
        return img, torch.tensor(numericalized)

# Collate: pad captions to batch max length
def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
    return imgs, padded, torch.tensor(lengths)

# ---------------------------
# 4) Encoder: pretrained ResNet -> feature vector
# ---------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=EMBED_SIZE, train_cnn=False):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # keep conv layers (feature map)
        self.feature_extractor = nn.Sequential(*modules)  # output: (B, 2048, H/32, W/32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.train_cnn = train_cnn
        if not train_cnn:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, images):
        # images: (B,3,H,W)
        feats = self.feature_extractor(images)           # (B,2048,h,w)
        pooled = self.adaptive_pool(feats).squeeze(-1).squeeze(-1)  # (B,2048)
        emb = self.fc(pooled)                            # (B,embed)
        emb = self.bn(emb)
        return emb  # (B, embed_size)

# ---------------------------
# 5) Transformer Decoder (uses nn.Transformer)
# We'll embed tokens and feed to TransformerDecoder
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=EMBED_SIZE, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads,
                                                   dim_feedforward=ffn_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def forward(self, tgt_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt_seq: (T, B) token ids
        # memory: (B, embed)  -> convert to (1, B, embed) and repeat across seq
        tgt = self.token_emb(tgt_seq) * math.sqrt(self.embed_size)  # (T,B,emb)
        tgt = self.pos_enc(tgt)
        memory = memory.unsqueeze(0)  # (1,B,emb)
        # transformer expects memory: (S, B, E), tgt: (T,B,E)
        out = self.transformer_decoder(tgt, memory,
                                       tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.fc_out(out)  # (T,B,vocab)
        return logits

# ---------------------------
# 6) Full model wrapper
# ---------------------------
class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def make_tgt_mask(self, size):
        # square subsequent mask
        mask = torch.triu(torch.ones(size,size), diagonal=1).bool().to(DEVICE)
        return mask

    def forward(self, images, captions):
        # images: (B,3,H,W), captions: (B, T)
        memory = self.encoder(images)  # (B, E)
        # prepare tgt: shift right (Transformer expects target with <sos> included)
        tgt_in = captions[:, :-1].transpose(0,1).to(DEVICE)  # (T-1, B)
        tgt_mask = self.make_tgt_mask(tgt_in.size(0))
        # padding mask: True where padding is (B, T)
        tgt_key_padding_mask = (tgt_in.transpose(0,1) == self.pad_idx).to(DEVICE)
        logits = self.decoder(tgt_in, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # logits: (T-1, B, V)
        return logits

# ---------------------------
# 7) Training utilities
# ---------------------------
def create_masks_and_targets(padded_captions, pad_idx):
    # returns inputs, targets, mask etc.
    # But model.forward handles masks; here we prepare targets for loss
    targets = padded_captions[:, 1:].transpose(0,1).to(DEVICE)  # (T-1, B)
    return targets

# ---------------------------
# 8) Main train loop (simplified)
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, vocab_size, pad_idx):
    model.train()
    total_loss = 0
    for images, captions, lengths in tqdm(dataloader):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images, captions)  # (T-1,B,V)
        targets = create_masks_and_targets(captions, pad_idx)  # (T-1,B)
        # reshape for loss
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.contiguous().view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------
# 9) Greedy inference
# ---------------------------
def greedy_generate(model, image, vocab, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)
        memory = model.encoder(image)  # (B=1,E)
        idxs = [vocab.stoi["<sos>"]]
        for _ in range(max_len-1):
            tgt = torch.tensor(idxs).unsqueeze(1).to(DEVICE)  # (T,1)
            tgt_mask = model.make_tgt_mask(tgt.size(0))
            out = model.decoder(tgt, memory, tgt_mask=tgt_mask)  # (T,1,V)
            next_token_logits = out[-1,0]  # (V,)
            next_id = next_token_logits.argmax().item()
            idxs.append(next_id)
            if next_id == vocab.stoi["<eos>"]:
                break
        # convert to words
        words = []
        for i in idxs[1:]:
            if i == vocab.stoi["<eos>"]: break
            if i == vocab.stoi["<pad>"]: continue
            words.append(vocab.itos.get(i, "<unk>"))
        return ' 
 10) Putting everything together (example usage)
def example_pipeline():
    # 0) Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 1) Load annotations (you must prepare annotations file first)
    # Example: annotations.json contains [{"image_path":"img1.jpg","caption":"a cat"} ...]
    annotations_file = "annotations.json"  # change to your file
    with open(annotations_file, 'r') as f:
        ann = json.load(f)
    all_captions = [a['caption'] for a in ann]

    # 2) Build vocab
    vocab = Vocab(freq_threshold=MIN_WORD_FREQ)
    vocab.build_vocabulary(all_captions)
    vocab_size = len(vocab.itos)
    print("Vocab size:", vocab_size)

    # 3) Dataset & DataLoader
    dataset = CaptionDataset(annotations_file, vocab, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 4) Model
    encoder = EncoderCNN(embed_size=EMBED_SIZE, train_cnn=False)
    decoder = DecoderTransformer(vocab_size=vocab_size, embed_size=EMBED_SIZE,
                                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                                 ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=MAX_LEN)
    pad_idx = vocab.stoi["<pad>"]
    model = ImageCaptioningModel(encoder, decoder, pad_idx).to(DEVICE)

    # 5) Loss & Optimizer (ignore pad in loss)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # 6) Train (small number of epochs for demo)
    EPOCHS = 10
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, vocab_size, pad_idx)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    # 7) Inference example (greedy)
    sample_img_path = ann[0]['image_path']
    img = Image.open(sample_img_path).convert('RGB')
    img_t = transform(img)
    caption = greedy_generate(model, img_t, vocab, max_len=MAX_LEN)
    print("Generated caption:", caption)

if __name__ == "__main__":
    example_pipeline()