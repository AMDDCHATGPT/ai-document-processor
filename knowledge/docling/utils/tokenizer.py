import tiktoken
from docling.chunking.base_chunker import BaseChunker

class OpenAITokenizerWrapper:
    """
    A wrapper for OpenAI's tiktoken tokenizer to work with docling
    """
    
    def __init__(self, model_name="cl100k_base"):
        """
        Initialize the tokenizer
        
        Args:
            model_name: The tiktoken model name (cl100k_base is used by gpt-4 and text-embedding models)
        """
        self.tokenizer = tiktoken.get_encoding(model_name)
    
    def encode(self, text):
        """
        Encode text to tokens
        
        Args:
            text: Input text string
            
        Returns:
            List of token ids
        """
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        """
        Decode tokens back to text
        
        Args:
            tokens: List of token ids
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(tokens)
    
    def count_tokens(self, text):
        """
        Count the number of tokens in a text
        
        Args:
            text: Input text string
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))