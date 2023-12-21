import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None): 
        bs,l=inputs_ids.size()
        inputs_ids=(inputs_ids.unsqueeze(1)).view(bs*1,l)
        position_idx=(position_idx.unsqueeze(1)).view(bs*1,l)  
        # 这种操作在处理多头注意力机制时常见，用于调整注意力掩码的形状以匹配输入数据的形状
        attn_mask=(attn_mask.unsqueeze(1)).view(bs*1,l,l)
        #embedding
        nodes_mask=position_idx.eq(0) # 等于0的是unk_token_id，表示的是数据流中的节点 
        token_mask=position_idx.ge(2) # 大于等于2的是源代码token
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        # 这是一种常见的归一化技术，用于将值转换为概率分布。1e-10 是一个很小的数，用于防止除以零的错误。
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]        
        
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
      
        

       
