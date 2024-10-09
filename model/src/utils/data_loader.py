import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class DataPreparer:
    def __init__(
        self,
        train_station_df,
        train_df,
        valid_df,
        tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased"),
        max_len=128,
        batch_size=32,
        test_df=None,
    ):
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.le_train_stations = LabelEncoder()
        self.le_train_stations.fit(train_station_df["LIBELLE"].unique())

        self.train_df = train_df()
        self.valid_df = valid_df()

        self.train_df['departure'] = self.le_train_stations.transform(train_df['departure'])
        self.train_df['arrival'] = self.le_train_stations.transform(train_df['arrival'])
        self.valid_df['departure'] = self.le_train_stations.transform(valid_df['departure'])
        self.valid_df['arrival'] = self.le_train_stations.transform(valid_df['arrival'])

        self.train_loader = self.__make_data_loader(self.train_df)
        self.val_loader = self.__make_data_loader(self.valid_df)

        if test_df is not None:
            self.test_df = test_df()
            self.test_df['departure'] = self.le_train_stations.transform(test_df['departure'])
            self.test_df['arrival'] = self.le_train_stations.transform(test_df['arrival'])
            self.test_loader = self.__make_data_loader(test_df)
        
    def __make_data_loader(self, df):
        dataset = DataEncoder(
            df["sentence"], df["departure"], df["arrival"], self.tokenizer, self.max_len)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        return loader


class DataEncoder(Dataset):
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

        
class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device = None):
        self.dl = dl
        if(device is None):
            self.device = self.get_default_device()
        else:
            self.device = device

    @staticmethod
    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @classmethod
    def to_device(cls, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [cls.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
