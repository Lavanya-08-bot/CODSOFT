import os
import re
import argparse
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchvision import models, transforms


# =========================
# Text Tokenization
# =========================

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z0-9']+", text)


# =========================
# Vocabulary Class
# =========================

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold

        self.itos = {
            0: "<pad>",
            1: "<start>",
            2: "<end>",
            3: "<unk>"
        }

        self.stoi = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3
        }

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi["<pad>"]

    @property
    def start_idx(self):
        return self.stoi["<start>"]

    @property
    def end_idx(self):
        return self.stoi["<end>"]

    @property
    def unk_idx(self):
        return self.stoi["<unk>"]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()

        for sentence in sentence_list:
            tokens = tokenize(sentence)
            frequencies.update(tokens)

        idx = len(self.itos)

        for word, count in sorted(frequencies.items()):
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text, max_len=40):
        tokens = tokenize(text)

        # Reserve space for <start> and <end>
        tokens = tokens[:max_len - 2]

        numericalized = [self.start_idx]

        for token in tokens:
            numericalized.append(self.stoi.get(token, self.unk_idx))

        numericalized.append(self.end_idx)

        return numericalized

    def to_dict(self):
        return {
            "freq_threshold": self.freq_threshold,
            "stoi": self.stoi,
            "itos": self.itos
        }

    @classmethod
    def from_dict(cls, data):
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.stoi = data["stoi"]
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        return vocab


# =========================
# Image-Caption Dataset
# =========================

class CaptionDataset(Dataset):
    def __init__(
        self,
        images_dir,
        captions_csv,
        vocab,
        transform=None,
        image_col="image",
        caption_col="caption",
        max_len=40
    ):
        self.images_dir = images_dir
        self.df = pd.read_csv(captions_csv)

        if image_col not in self.df.columns:
            raise ValueError(f"Column '{image_col}' not found in CSV.")

        if caption_col not in self.df.columns:
            raise ValueError(f"Column '{caption_col}' not found in CSV.")

        self.images = self.df[image_col].astype(str).values
        self.captions = self.df[caption_col].astype(str).values

        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_name = self.images[index]
        caption = self.captions[index]

        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_ids = self.vocab.numericalize(caption, self.max_len)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor


# =========================
# Collate Function
# =========================

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images, dim=0)

        captions = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return images, captions


# =========================
# Image Transform
# =========================

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# ResNet50 Helper
# =========================

def get_resnet50():
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)

    return model


# =========================
# CNN Encoder
# =========================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, train_cnn=False, dropout=0.3):
        super().__init__()

        self.train_cnn = train_cnn

        resnet = get_resnet50()

        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        for param in resnet.parameters():
            param.requires_grad = train_cnn

        self.resnet = resnet

        self.fc = nn.Linear(in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        if self.train_cnn:
            features = self.resnet(images)
        else:
            self.resnet.eval()
            with torch.no_grad():
                features = self.resnet(images)

        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)

        return features


# =========================
# LSTM Decoder
# =========================

class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.init_h = nn.Linear(embed_size, hidden_size * num_layers)
        self.init_c = nn.Linear(embed_size, hidden_size * num_layers)

    def init_hidden_state(self, image_features):
        batch_size = image_features.size(0)

        h = self.init_h(image_features)
        c = self.init_c(image_features)

        h = h.view(batch_size, self.num_layers, self.hidden_size)
        c = c.view(batch_size, self.num_layers, self.hidden_size)

        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()

        return h, c

    def forward(self, image_features, captions):
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)

        states = self.init_hidden_state(image_features)

        outputs, _ = self.lstm(embeddings, states)
        predictions = self.fc(outputs)

        return predictions

    def greedy_search(self, image_features, start_idx, end_idx, max_len=30):
        generated_tokens = []

        states = self.init_hidden_state(image_features)

        input_token = torch.tensor(
            [[start_idx]],
            dtype=torch.long,
            device=image_features.device
        )

        for _ in range(max_len):
            embedding = self.embedding(input_token)

            output, states = self.lstm(embedding, states)

            logits = self.fc(output.squeeze(1))

            predicted_token = logits.argmax(dim=1)

            token_id = predicted_token.item()

            if token_id == end_idx:
                break

            generated_tokens.append(token_id)

            input_token = predicted_token.unsqueeze(1)

        return generated_tokens


# =========================
# Training Function
# =========================

def train(args):
    device = torch.device(args.device)

    df = pd.read_csv(args.captions_csv)

    if args.image_col not in df.columns:
        raise ValueError(f"CSV must contain column '{args.image_col}'")

    if args.caption_col not in df.columns:
        raise ValueError(f"CSV must contain column '{args.caption_col}'")

    print("Building vocabulary...")
    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(df[args.caption_col].astype(str).tolist())

    print(f"Vocabulary size: {len(vocab)}")

    transform = get_transform(train=True)

    dataset = CaptionDataset(
        images_dir=args.images_dir,
        captions_csv=args.captions_csv,
        vocab=vocab,
        transform=transform,
        image_col=args.image_col,
        caption_col=args.caption_col,
        max_len=args.max_len
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MyCollate(vocab.pad_idx)
    )

    encoder = EncoderCNN(
        embed_size=args.embed_size,
        train_cnn=args.train_cnn
    ).to(device)

    decoder = DecoderRNN(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    trainable_params = [
        p for p in list(encoder.parameters()) + list(decoder.parameters())
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()

        total_loss = 0

        progress_bar = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]")

        for images, captions in progress_bar:
            images = images.to(device)
            captions = captions.to(device)

            # captions[:, :-1] = input tokens
            # captions[:, 1:] = target tokens
            caption_inputs = captions[:, :-1]
            caption_targets = captions[:, 1:]

            image_features = encoder(images)

            outputs = decoder(image_features, caption_inputs)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caption_targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "vocab": vocab.to_dict(),
            "model_config": {
                "embed_size": args.embed_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers
            }
        }

        torch.save(checkpoint, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")


# =========================
# Load Model Function
# =========================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab = Vocabulary.from_dict(checkpoint["vocab"])

    config = checkpoint["model_config"]

    encoder = EncoderCNN(
        embed_size=config["embed_size"],
        train_cnn=False
    ).to(device)

    decoder = DecoderRNN(
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        vocab_size=len(vocab),
        num_layers=config["num_layers"]
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab


# =========================
# Caption Generation
# =========================

def generate_caption(args):
    device = torch.device(args.device)

    encoder, decoder, vocab = load_model(args.checkpoint, device)

    transform = get_transform(train=False)

    image = Image.open(args.image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = encoder(image)

        token_ids = decoder.greedy_search(
            image_features=image_features,
            start_idx=vocab.start_idx,
            end_idx=vocab.end_idx,
            max_len=args.generate_max_len
        )

    words = []

    for token_id in token_ids:
        word = vocab.itos.get(token_id, "<unk>")

        if word not in ["<pad>", "<start>", "<end>"]:
            words.append(word)

    caption = " ".join(words)

    print("\nGenerated Caption:")
    print(caption)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "caption"], required=True)

    parser.add_argument("--images_dir", type=str, default="data/images")
    parser.add_argument("--captions_csv", type=str, default="data/captions.csv")

    parser.add_argument("--image_col", type=str, default="image")
    parser.add_argument("--caption_col", type=str, default="caption")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/image_captioning.pth")

    parser.add_argument("--image", type=str, default=None)

    parser.add_argument("--freq_threshold", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--generate_max_len", type=int, default=30)

    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--train_cnn", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "caption":
        if args.image is None:
            raise ValueError("Please provide --image path for caption mode.")

        generate_caption(args)


if __name__ == "__main__":
    main()