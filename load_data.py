import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def preprocess_nl_query(nl_query: str) -> str:
    return "translate to SQL: " + nl_query

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        
        # Get BOS token ID (<extra_id_0>) for decoder, with fallback to other extra_id tokens if needed
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if self.bos_token_id == self.tokenizer.unk_token_id:
            for i in range(1, 100):
                token_id = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
                if token_id != self.tokenizer.unk_token_id:
                    self.bos_token_id = token_id
                    break
            else:
                self.bos_token_id = self.tokenizer.pad_token_id
        
        # Load NL queries
        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl_texts = load_lines(nl_path)
        
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql_texts = load_lines(sql_path)
            assert len(self.nl_texts) == len(self.sql_texts), "NL and SQL files must have same length"
        else:
            self.sql_texts = None
        
        self.processed_nl_texts = []
        for nl in self.nl_texts:
            processed_nl = preprocess_nl_query(nl)
            self.processed_nl_texts.append(processed_nl)

    def __len__(self):
        return len(self.nl_texts)

    def __getitem__(self, idx):
        nl_text = self.processed_nl_texts[idx]
        
        encoder_inputs = self.tokenizer(
            nl_text,
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        encoder_ids = encoder_inputs["input_ids"].squeeze(0)
        
        if self.split == "test":
            initial_decoder_token = torch.tensor([self.bos_token_id])
            return encoder_ids, initial_decoder_token
        else:
            sql_text = self.sql_texts[idx]
            
            decoder_targets = self.tokenizer(
                sql_text,
                max_length=512,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            decoder_target_ids = decoder_targets["input_ids"].squeeze(0)
            
            decoder_input_ids = torch.cat([
                torch.tensor([self.bos_token_id]),
                decoder_target_ids[:-1]
            ])
            
            initial_decoder_token = torch.tensor([self.bos_token_id])
            
            return encoder_ids, decoder_input_ids, decoder_target_ids, initial_decoder_token

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item[0] for item in batch]
    decoder_input_ids_list = [item[1] for item in batch]
    decoder_target_ids_list = [item[2] for item in batch]
    initial_decoder_tokens_list = [item[3] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_target_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    initial_decoder_inputs = torch.stack(initial_decoder_tokens_list)
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item[0] for item in batch]
    initial_decoder_tokens_list = [item[1] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    initial_decoder_inputs = torch.stack(initial_decoder_tokens_list)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    """
    Load raw NL and SQL strings for prompting-based experiments.

    Returns:
        train_x: list of train NL queries (strings)
        train_y: list of train SQL queries (strings)
        dev_x:   list of dev NL queries (strings)
        dev_y:   list of dev SQL queries (strings)
        test_x:  list of test NL queries (strings, no SQL labels)
    """
    # Train split
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))

    # Dev split
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))

    # Test split (no SQL ground truth)
    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    return train_x, train_y, dev_x, dev_y, test_x