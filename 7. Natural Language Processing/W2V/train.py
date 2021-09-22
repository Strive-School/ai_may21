import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_handler as dh
import numpy as np

dataset, word2idx, vocab_size = dh.get_dataset()

print(dataset[1])


def input_layer(word_idx):
	x = torch.zeros(vocab_size)
	x[word_idx] = 1.0
	return x

x = input_layer(dataset[0][0])
print(x)
print(x.shape)
print(vocab_size)

def train(n_epochs = 1, lr = 0.01, embedding_size = 10):
	
    W1 = Variable( torch.rand(vocab_size, embedding_size).float(), requires_grad = True )
    W2 = Variable( torch.rand(embedding_size, vocab_size).float(), requires_grad = True)

    for epoch in range(n_epochs):
        loss_val = 0
        
        for data, target in dataset:
        
            x = Variable(input_layer(data)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())
            
            z1 = torch.matmul(x,W1)
            z2 = torch.matmul(z1,W2)
            
            log_softmax = F.log_softmax(z2, dim=0)
            loss = F.nll_loss(log_softmax.view(1,-1), y_true)

            loss.backward()
            loss_val += loss.item()
    
            W1.data -= lr * W1.grad.data
            W2.data -= lr * W2.grad.data
    
            W1.grad.data.zero_()
            W2.grad.data.zero_()
            
        if epoch % 10 == 0:    
            print(f'Loss at epoch {epoch}: {loss_val/len(dataset)}')
    return W1

w1 = train()


input = input_layer( word2idx["you"] )


embedding = torch.matmul(input,w1 )

print(embedding)
