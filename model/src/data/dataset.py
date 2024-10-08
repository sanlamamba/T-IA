import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, sentences, departure, arrival, tokenizer, max_len):
        self.sentences = sentences
        self.departure = departure
        self.arrival = arrival
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'departure': torch.tensor(self.departure[idx], dtype=torch.long),
            'arrival': torch.tensor(self.arrival[idx], dtype=torch.long)
        }