class Config:
    VOCAB_SIZE = 10000
    MIN_FREQUENCY = 1
    BATCH_SIZE = 64
    MAX_SEQUENCE_LENGTH = 256
    D_MODEL = 256
    NUM_LAYERS = 4
    NUM_HEADS = 8 
    HIDDEN_DIM = 1024
    DROPOUT = 0.1
    LEARNING_RATE = 5e-4
    warmup_ratio = 0.05
    EPOCHS = 20