import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import MLP, get_dataset_from_model


class DynamicsLearner:
    """
    Dynamics Learning model with differentiable predictor.
    """
    def __init__(self, dl_config, seed=0):
        torch.manual_seed(seed)
        self.extract_config(dl_config)
        self.model = MLP(input_dim=self.input_dim, hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)    

    def extract_config(self, config):
        """
        Extract config dictionary and set all values as class attributes.
        """
        for key, value in config.items():
            setattr(self, key, value)
    
    def predict(self, state, action):
        """
        Predict next states for a given state and action.
        """
        self.model.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = torch.LongTensor(action).to(self.device)
            input = torch.cat((state, action), dim=1)
            return self.model(input)
        
    def batch_predict(self, X, A):
        """
        Predict next states for a batch of states and actions.
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            A = torch.LongTensor(A).to(self.device)
            batch_input = torch.cat((X, A), dim=1)
            return self.model(batch_input)
    
    def fit(self, state, action, target):
        """
        Fit the model to a single state plus action and target.
        """
        self.model.train()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        input = torch.cat((state, action), dim=1)
        target = torch.FloatTensor(target).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
    
    def batch_fit(self, X, A, Xp):
        """
        Fit the model to a batch of states plus actions and targets.
        """
        self.model.train()
        X = torch.FloatTensor(X).to(self.device)
        A = torch.LongTensor(A).to(self.device).reshape(-1, 1)
        input = torch.cat((X, A), dim=1)
        Xp = torch.FloatTensor(Xp).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.criterion(output, Xp)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, filename):
        """
        Save model weights to a file.
        """
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename):
        """
        Load model weights from a file.
        """
        self.model.load_state_dict(torch.load(filename, weights_only=True))
        self.model.eval()


def train_dynamics_learner(dl_config, agent, env_config):
    # Collect training dataset for the dynamics learner
    X_train, A_train, Xp_train = get_dataset_from_model(agent, env_config, episodes=dl_config["train_data_episodes"])
    np.savez_compressed("data/datasets/dl_data_train.npz", X=X_train, A=A_train, Xp=Xp_train)
    X_test, A_test, Xp_test = get_dataset_from_model(agent, env_config, episodes=dl_config["test_data_episodes"])
    np.savez_compressed("data/datasets/dl_data_test.npz", X=X_test, A=A_test, Xp=Xp_test)

    X_train = torch.from_numpy(X_train).float()
    A_train = torch.from_numpy(A_train).long()
    Xp_train = torch.from_numpy(Xp_train).float()
    train_dataset = TensorDataset(X_train, A_train, Xp_train)
    train_loader = DataLoader(train_dataset, batch_size=dl_config["batch_size"], shuffle=True)
    
    # Train dynamics learner
    dl = DynamicsLearner(dl_config)
    for epoch in tqdm(range(dl_config["max_epochs"])):
        total_epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
                X_batch, A_batch, Xp_batch = batch
                batch_loss = dl.batch_fit(X_batch, A_batch, Xp_batch)
                total_epoch_loss += batch_loss
                num_batches += 1

        # Average batch loss for this epoch
        avg_epoch_loss = total_epoch_loss / num_batches

        if epoch % 20 == 0:
            print(f"DL epoch {epoch}: Avgerage Loss {avg_epoch_loss:.6f}")

    # Evaluate the dynamics learner
    with torch.no_grad():
        X_test = torch.from_numpy(X_test).float()
        A_test = torch.from_numpy(A_test).long().reshape(-1, 1)
        test_input = torch.cat((X_test, A_test), dim=1).to(dl.device)
        test_output = dl.model(test_input)
        test_target = torch.from_numpy(Xp_test).float().to(dl.device)
        test_loss = dl.criterion(test_output, test_target)
        print(f"DL test loss: {test_loss.item():.6f}")

    # Collect data from the dynamics learner for data attribution
    target_input = torch.cat((X_test, A_test), dim=1).to(dl.device)
    target_output = dl.model(target_input)
    np.savez_compressed("data/datasets/dl_target_rollouts.npz",
                        X=X_test.cpu().numpy(),
                        A=A_test.cpu().numpy(),
                        Xp=target_output.detach().cpu().numpy())

    # Save the dynamics learner
    dl.save_model(f"data/models/dynamics_learner.pt")
    return dl, test_loss.item()
