
from pathlib import Path
from data.dataset import WikiText2
from data.preprocessing import TextPreprocessor, clean_text
from data.tokenizer import Tokenizer
from data.vocabulary import Vocabulary
from model.mini_gpt import MiniGPT
from model.transformer_head import TransformerHead
from training.config import Config
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from training.trainer import Trainer 

def load_text_file(file_path):
    """
    Read a text file and return a list of lines.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    """
    Main training pipeline.
    """
    root_dir = Path(__file__).parent

    # 1. Load dataset
    train_path = root_dir / "data" / "dataset" / "train.txt"
    test_path = root_dir / "data" / "dataset" / "test.txt"
    
    train_texts = load_text_file(train_path)
    test_texts = load_text_file(test_path)
   
    # 2. Initialize tokenizer
    tokenizer = Tokenizer()
    
    # 3. Build vocabulary
    vocab = Vocabulary(Config.VOCAB_SIZE, Config.MIN_FREQUENCY)
    vocab.build_vocabulary(train_texts, tokenizer)
    
    
    sample = train_texts[3]
    print("\nRAW:")
    print(sample)

    cleaned = clean_text(sample)
    print("\nCLEANED:")
    print(cleaned)
    
    print("\nCLEANED Max:")
    print(cleaned[:Config.MAX_SEQUENCE_LENGTH])

    tokens = tokenizer.tokenize(cleaned[:Config.MAX_SEQUENCE_LENGTH])
    print("\nTOKENS:")
    print(tokens)

    encoded = vocab.encode(tokens)
    print("\nENCODED:")
    print(encoded)

    decoded = vocab.decode(encoded)
    print("\nDECODED:")
    print(decoded)
    
    print("\nVOCAB SIZE:")
    print(len(vocab))
    
    
    # 4. Create preprocessor
    preprocessor = TextPreprocessor(tokenizer, vocab)
    
    # 5. Create datasets
    train_data = WikiText2(train_texts, preprocessor, Config.MAX_SEQUENCE_LENGTH)
    valid_data = WikiText2(test_texts, preprocessor, Config.MAX_SEQUENCE_LENGTH)
    
    # 6. Create dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 7. Initialize model
    mini_gpt = MiniGPT(len(vocab), Config.D_MODEL,Config.NUM_LAYERS,Config.NUM_HEADS,Config.HIDDEN_DIM,Config.DROPOUT, Config.MAX_SEQUENCE_LENGTH)
    transformer_head = TransformerHead(mini_gpt, Config.D_MODEL, len(vocab))
    
    # 8. Initialize optimizer and loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(params=transformer_head.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)
    
    # 9. Initialize trainer
    trainer = Trainer(transformer_head, train_loader, valid_loader, optimizer, criterion, Config.LEARNING_RATE, Config.warmup_ratio)
    
    # 10. Start training
    trainer.train(Config.EPOCHS)
    
if __name__ == "__main__":
    main()