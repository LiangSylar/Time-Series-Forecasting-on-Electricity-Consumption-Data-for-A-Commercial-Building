import numpy as np
import csv
import torch
import math
from torchvision import models
import matplotlib.pyplot as plt
n_features = 8 # this is number of parallel inputs
n_timesteps = 96 # this is number of timesteps

training_csv = 'CL_imputed.csv'
with open(p,encoding = 'utf-8') as f:
    data = np.loadtxt(f,str,delimiter = ",",skiprows = 1,usecols = (1,2,3,4,8,9,10,7))
data = data.astype('float')
y_std = np.std(data[:,-1])
for i in range(data.shape[1]):
    data[:,i] = (data[:,i]-np.mean(data[:-672,i]))/np.std(data[:-672,i])
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix >= len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 64 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        #self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        self.resmodel = models.resnet18(pretrained=False)
        num_ftrs = self.resmodel.fc.in_features
        self.resmodel.fc = torch.nn.Linear(num_ftrs, 1)
        self.resmodel.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #for name, p in self.l_lstm.named_parameters():
        #    if "weight" in name:
        #        torch.nn.init.orthogonal_(p)
        #    elif "bias" in name:
        #        torch.nn.init.constant_(p, 0)
    
    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,1,n_timesteps,self.n_hidden)
        return self.resmodel(x)
        
X, Y = split_sequences(data, n_timesteps)
#X = np.expand_dims(X, axis=1)
print(data.shape)
print(X.shape)
print(Y.shape)
print(y_std)

train_episodes = 300
batch_size =64
lr_rate = 0.001

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr_rate)
start = 20000
X_train = X[start:X.shape[0]-2500]
X_test = X[X.shape[0]-2500:]
X_test[:,:,-1] = 0
y_train = Y[start:X.shape[0]-2500]
y_test = Y[X.shape[0]-2500:]
#train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
#test_loader = torch.utils.data.DataLoader(data_test, shuffle=False, num_workers=4)

mv_net = mv_net.to(device)
criterion = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr_rate)
best_loss = 9999.0
best_ep = -1
pred_result = np.ndarray([X_test.shape[0]])
for t in range(train_episodes):
    if((t+1)%100 == 0):
        lr_rate = lr_rate/3
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_rate
    train_loss = []
    mv_net.train()
    for b in range(0,len(X_train),batch_size):
        inpt = X_train[b:b+batch_size,:,:]
        target = y_train[b:b+batch_size]    
        x_batch = torch.tensor(inpt,dtype=torch.float32).to(device)   
        y_batch = torch.tensor(target,dtype=torch.float32).to(device)
        mv_net.init_hidden(x_batch.size(0), device)
        output = mv_net(x_batch)
        loss = criterion(output.view(-1), y_batch)  
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
    #scheduler.step()
    test_loss = []
    mv_net.init_hidden(1, device)
    mv_net.eval()
    for b in range(0,len(X_test)):
        #print(X_test.shape)
        inpt = X_test[b:b+1,:,:]
        target = y_test[b:b+1]    
        
        x_batch = torch.tensor(inpt,dtype=torch.float32).to(device)   
        y_batch = torch.tensor(target,dtype=torch.float32).to(device)
    
        output_test = mv_net(x_batch)
        loss = criterion(output_test.view(-1), y_batch)  
        test_loss.append(loss.item())
        for i in range(n_timesteps):
            if(b+i+1<X_test.shape[0]):
                X_test[b+1+i,-i,-1] = output_test.view(-1).cpu().detach().numpy()
        pred_result[b] = output_test.view(-1).cpu().detach().numpy()
    plt.figure(figsize=(30,5))
    plt.plot(range(len(pred_result)), pred_result)
    plt.plot(range(len(y_test)), y_test)
    #plt.show()
    plt.savefig('bs64/ep{}_bs{}_lr{}_layer{}_hidden{}.jpg'.format(t,batch_size,lr_rate,2,64))
    plt.close()
    if (math.sqrt(np.mean(test_loss))*y_std<best_loss):
        best_loss = math.sqrt(np.mean(test_loss))*y_std
        best_ep = t
    print('step : ' , t , 'train_rmse : ' , math.sqrt(np.mean(train_loss))*y_std, 'test_rmse : ' , math.sqrt(np.mean(test_loss))*y_std, 'best_loss : ' , best_loss, 'test_ep : ' , best_ep)