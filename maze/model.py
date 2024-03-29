import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size,device=device)
        self.linear2 = nn.Linear(hidden_size, output_size,device=device)
        self.device = device

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        pt.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = self.model.device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        
        state = pt.tensor(state, dtype=pt.float,device=self.device)
        next_state = pt.tensor(next_state, dtype=pt.float, device=self.device)
        action = pt.tensor(action, dtype=pt.long, device=self.device)
        reward = pt.tensor(reward, dtype=pt.float, device=self.device)

        if len(state.shape) == 1: # short term memory check (1,x)
            state = pt.unsqueeze(state, 0)
            next_state = pt.unsqueeze(next_state, 0)
            action = pt.unsqueeze(action, 0)
            reward = pt.unsqueeze(reward, 0)
            done = (done, )

        # predict Q values based on current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Q_new = r + y * max(future_Qs)
                Q_new = reward[idx] + self.gamma * pt.max(self.model(next_state[idx]))

            target[idx][pt.argmax(action[idx]).item()] = Q_new
    
        # optimize with new q vals
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()