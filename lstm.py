import numpy as np
import csv
import torch
import math
p = 'method2_CoolingLoad18months.csv'
with open(p,encoding = 'utf-8') as f:
    data = np.loadtxt(f,str,delimiter = ",",skiprows = 1,usecols = (1,2,3,4,7))
data = data.astype('float')

for i in range(data.shape[1]-1):
    data[:,i] = (data[:,i]-np.mean(data[:,i]))/np.std(data[:,i])

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

print(data.shape)
print(data[:1])
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 256 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)
n_features = 4 # this is number of parallel inputs
n_timesteps = 96 # this is number of timesteps

# convert dataset into input/output
X, y = split_sequences(data, n_timesteps)
print(X.shape, y.shape)
X_train = X[:int(X.shape[0]*0.8),:,:]
X_test = X[int(X.shape[0]*0.8+1):,:,:]
y_train = y[:int(y.shape[0]*0.8)]
y_test = y[int(y.shape[0]*0.8+1):]
print(X_train.shape, y_train.shape)
# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.SGD(mv_net.parameters(), lr=1e-7)

train_episodes = 100
batch_size =128
mv_net.train()
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(X_train), eta_min=0, last_epoch=-1)
for t in range(train_episodes):
    for b in range(0,len(X_train),batch_size):
        inpt = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]    
        
        x_batch = torch.tensor(inpt,dtype=torch.float32)    
        y_batch = torch.tensor(target,dtype=torch.float32)
    
        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch)
        loss = criterion(output.view(-1), y_batch)  
        
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
    #scheduler.step()
    mv_net.eval()
    x_batch_test = torch.tensor(X_test,dtype=torch.float32)    
    y_batch_test = torch.tensor(y_test,dtype=torch.float32)
    mv_net.init_hidden(x_batch_test.size(0))
    output_test = mv_net(x_batch_test)
    loss_test = criterion(output_test.view(-1), y_batch_test)
    mv_net.train()
    print('step : ' , t , 'train_rmse : ' , math.sqrt(loss.item()), 'test_rmse : ' , math.sqrt(loss_test.item()))
