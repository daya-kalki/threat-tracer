import pandas as pd
import numpy as np
import torch
import transformers
import tokenizers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, LineByLineTextDataset, BertModel, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import logging
logging.basicConfig(level=logging.ERROR)

train_data = pd.read_csv('../input/query-wellformedness/train.tsv', sep='\t', header=None)
val_data = pd.read_csv('../input/query-wellformedness/dev.tsv', sep='\t', header=None)
test_data = pd.read_csv('../input/query-wellformedness/test.tsv', sep='\t', header=None)

train_data.columns = ['query', 'label']
val_data.columns = ['query', 'label']
test_data.columns = ['query', 'label']

train_data['label'] = [1 if label>=0.8 else 0 for label in train_data['label']]
val_data['label'] = [1 if label>=0.8 else 0 for label in val_data['label']]
test_data['label'] = [1 if label>=0.8 else 0 for label in test_data['label']]

pretraining_data = train_data['query'].tolist() + val_data['query'].tolist() + test_data['query'].tolist()

with open('/kaggle/working/pretraining_data.txt', 'w') as f:
    for sent in pretraining_data:
        f.write("%s\n" % sent)

bwpt = tokenizers.BertWordPieceTokenizer()
 
filepath = "/kaggle/working/pretraining_data.txt"

bwpt.train(
    files=[filepath],
    vocab_size=50000,
    min_frequency=3,
    limit_alphabet=1000
)

bwpt.save_model('/kaggle/working/')

vocab_file_dir = '/kaggle/working/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = '/kaggle/working/pretraining_data.txt',
    block_size = 128
)

print('No. of lines: ', len(dataset))

config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512
)
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir='/kaggle/working/',
    overwrite_output_dir=True,
    num_train_epochs=7,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
#     prediction_loss_only=True,
)

trainer.train()
trainer.save_model('/kaggle/working/')

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
LEARNING_RATE = 1e-05

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)

class QueryData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe['query']
        self.targets = dataframe['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

print("Train Dataset: {}".format(train_data.shape))
print("Validation Dataset: {}".format(val_data.shape))
print("Test Dataset: {}".format(test_data.shape))

training_set = QueryData(train_data, tokenizer, MAX_LEN)
val_set = QueryData(val_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **test_params)

class BertClass(torch.nn.Module):
    def __init__(self):
        super(BertClass, self).__init__()
        self.l1 = BertModel.from_pretrained("pytorch_model.bin")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = BertClass
()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(epoch, training_loader):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%500==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 500 steps: {loss_step}")
            print(f"Training Accuracy per 500 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 
EPOCHS = 2
for epoch in range(EPOCHS):
    train(epoch, training_loader)

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


acc = valid(model, val_loader)
print("Accuracy on validation data = %0.2f%%" % acc)

class QueryData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['query']
#         self.targets = self.data.citation_influence_label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# %%
test_data.head(2)

# %%
data_to_test = QueryData(test_data[['query']], tokenizer, MAX_LEN)

test_params = {'batch_size': 4,
                'shuffle': False,
                'num_workers': 0
                }

testing_loader_f = DataLoader(data_to_test, **test_params)

# %%
def test(model, testing_loader):
    res = []
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
#             targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            big_val, big_idx = torch.max(outputs, dim=1)
            res.extend(big_idx.tolist())
    
    return res

# %%
res = test(model, testing_loader_f)

# %%
test_data['label'] = [1 if label>=0.8 else 0 for label in test_data['label']]

# %%
correct = [1 if pred==lab else 0 for pred, lab in zip(res, test_data['label'].tolist())]
print('accuracy on test set is - ', sum(correct)/len(test_data['label'].tolist())*100, '%')

# %%
output_model_file = 'pytorch_bert_pretrained.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')
print('This tutorial is completed')


