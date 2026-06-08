from torch.utils.data import Dataset
import torch

class WikiText2(Dataset):
    def __init__(self, texts, preprocessor, max_length=256):
        self.texts = texts
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.samples = []
        
        for text in texts:
            tokens = self.preprocessor.process(text)
        
            # create chunks
            for start in range(0, len(tokens), self.max_length):
                chunk = tokens[start:start + self.max_length + 1]
                if len(chunk) < 2:
                        continue
                self.samples.append(chunk)
            
        print("Original documents:", len(texts))
        print("Generated samples:", len(self.samples))
        
    def __len__(self):
        """
        Return: length of the text 
        """
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Return: 
        input: actual text for train
        target: actual text text shifted right
        attention_mask: text attention mask
        """
        # Get sample
        chunk = self.samples[index]
        chunk = self.preprocessor.pad_sequence(sequence=chunk, max_length=self.max_length + 1)
        
        # Create input and target
        input_ids = torch.tensor(data=chunk[:-1], dtype=torch.long)
        target_ids  = torch.tensor(data=chunk[1:], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        
        return input_ids, target_ids, attention_mask