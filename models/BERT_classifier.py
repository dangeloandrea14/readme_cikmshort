import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased") 
    
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, n_classes)

        self.relu = nn.ReLU()
        self.last_layer = self.fc2

    def forward(self, X):

        input_ids = X[:, 0, :].long()  
        attention_mask = X[:, 1, :].long()  


        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        x = outputs.last_hidden_state[:, 0, :]  
        x = self.relu(self.fc1(x))
        
        intermediate_output = x  
        x = self.fc2(x)
        
        return intermediate_output, x
