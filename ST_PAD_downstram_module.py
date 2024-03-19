import torch
import torch.nn as nn

class EnvEncoder(nn.Module):
    def __init__(self):
        super(EnvEncoder, self).__init__()
        # Define the architecture here. Example:
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=C, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size) # `feature_size` is the size of the environmental feature vector
        )
        
    def forward(self, x):
        return self.encoder(x)


class TimeEmbedding(nn.Module):
    def __init__(self, max_time, feature_size):
        super(TimeEmbedding, self).__init__()
        self.time_embedding = nn.Embedding(max_time, feature_size)
        
    def forward(self, t):
        return self.time_embedding(t)
        
class DiffusionDDSD(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, num_steps):
        super(DiffusionDDSD, self).__init__()
        # feature_dim: Dimension of the combined input features (environmental features + time embedding + z_next)
        # hidden_dim: Dimension of hidden layers in the network
        # output_dim: Dimension of the output, typically the size of the original input data
        # num_steps: Number of diffusion steps to simulate
        
        self.num_steps = num_steps
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Beta schedule for noise levels, can be learned or predefined
        self.beta = torch.linspace(0.01, 0.1, steps=num_steps)

    def forward(self, z_next, environmental_features, time_embedding, t):
        # t: Current diffusion time step
        combined_input = torch.cat([z_next, environmental_features, time_embedding], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        noise_pred = self.fc3(x)  # Predicting the noise
        

        beta_t = self.beta[t]
        z_pred = (z_next - beta_t * noise_pred) / (1 - beta_t)  # Simplified denoising step
        
        return z_pred

    def reverse_diffusion(self, z_noisy, environmental_features, time_embedding):
        for t in reversed(range(self.num_steps)):
            z_noisy = self.forward(z_noisy, environmental_features, time_embedding, t)
        return z_noisy

class DDSD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DDSD, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, z_next, environmental_features, time_embedding):
        combined_input = torch.cat([z_next, environmental_features, time_embedding], dim=1)
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        
        return x
