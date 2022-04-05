import torch
import torch.nn as nn


class KnowledgeEmbedder(nn.Module):
    def __init__(self, tokenizer,
            checkpoint,
            max_q_len, max_k_len, max_k_num) -> None:
        """
        Args:
            tokenizer (nn.Module): pre-trained tokenizer
            checkpoint (str): pre-trained checkpoint name
            max_q_len (int): max question length
            max_k_len (int): max knowledge length
            max_k_num (int): max knowledge sentence numbers
        """
        super().__init__()
        self.tokenizer = tokenizer.from_pretrained(checkpoint)
        self.max_q_len = max_q_len
        self.max_k_len = max_k_len
        self.max_k_num = max_k_num
        self.max_len = max_q_len + max_k_len

    def forward(self, knowledge, question):
        """
        Args:
            knowledge (list): list of topk knowledge
            question (list): list of question words (same size with batch)
        """
        text = []
        for words_q, topk_knowledge in zip(question, knowledge):
            for words_k in topk_knowledge:
                text.append(' '.join(words_k.split()[:self.max_k_len]
                    + words_q.split()[:self.max_q_len]))
        inputs_text = self.tokenizer(text, 
            return_tensors='pt',
            max_length=self.max_len, padding='max_length', truncation=True)
        
        if torch.cuda.is_available():
            inputs_text = {k: v.cuda() for k, v in inputs_text.items()}

        return inputs_text
        