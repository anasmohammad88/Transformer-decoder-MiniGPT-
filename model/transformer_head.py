from torch import nn

class TransformerHead(nn.Module):
    def __init__(self, mini_gpt, d_model, vocab_size):
        super().__init__()
        self.mini_gpt = mini_gpt
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)
        self.classifier.weight = self.mini_gpt.embedding.embedding.weight
        
    def forward(self, x, attention_mask):
        x = self.mini_gpt(x, attention_mask)
        logits = self.classifier(x)
        return logits