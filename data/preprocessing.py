import re

class TextPreprocessor:
    def __init__(self, tokenizer, vocabulary):
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.pad_token = "<PAD>"

    def process(self, text: str):
        """
        Full preprocessing pipeline.
        """
        # Clean text
        text = clean_text(text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Add BOS and EOS tokens 
        tokens = [self.vocabulary.BOS_TOKEN] + tokens + [self.vocabulary.EOS_TOKEN ] 

        # Convert tokens to vocabulary indices
        encoded_tokens = self.vocabulary.encode(tokens)
        
        return encoded_tokens
    
    def pad_sequence(self, sequence, max_length):
        """
        Pad or truncate sequence.
        """

        # Truncate if too long
        if len(sequence) > max_length:
            sequence = sequence[:max_length - 1] + [self.vocabulary.word_to_index[self.vocabulary.EOS_TOKEN ]]

        # Pad if too short
        if len(sequence) < max_length:
            padding_length = max_length - len(sequence)
            sequence += [0] * padding_length

        return sequence

def clean_text(text: str):
    """
    Clean raw text.
    """
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text