import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, num_genes):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, 256),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, num_genes)
        )

    def forward(self, x, t):
        return self.net(x)

# Define the DDPM model
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=None, device=None, num_genes=None):
        super(MyDDPM, self).__init__()
        self.network = network.to(device)
        self.n_steps = n_steps
        self.device = device
        self.num_genes = num_genes

        # Create the beta schedule for variance
        self.beta_schedule = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha_schedule = 1. - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)

    def q_sample(self, x0, t, noise=None):
        # Sample from the diffusion process at time t given data x0.
        if noise is None:
            noise = torch.randn_like(x0)


        alpha_bar_t = self.alpha_bar_schedule[t]
        if alpha_bar_t.ndim == 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(0)
    
        alpha_bar_t = alpha_bar_t.unsqueeze(1) if alpha_bar_t.ndim == 1 else alpha_bar_t
        
         
     
        # Calculate the mean for the q distribution
        mean = (alpha_bar_t.sqrt() * x0)
        # Calculate the variance for the q distribution
        variance = (1 - alpha_bar_t).sqrt()

        # Return the noised data and the noise
        return mean + variance * noise, noise

    def p_losses(self, x0, t, noise=None):
        # Compute the losses for the p distribution (reversed process)
        x_noisy, noise = self.q_sample(x0, t, noise)
        # Predict the noise using the model
        noise_pred = self.network(x_noisy, t)

        # Compute the loss as the mean squared error between the predicted and actual noise
        loss = torch.mean((noise - noise_pred) ** 2)
        return loss

    def p_sample(self, x, t, cell_type, noise=None):
        # Sample from the model's distribution p at time t.
        if torch.all(t == 0):
            return x

        if noise is None:
            noise = torch.randn_like(x)

        # Compute the previous mean using the predicted noise
        noise_pred = self.network(x, cell_type, t)
        
        t_unsq = t.unsqueeze(1)  # Add an extra dimension for broadcasting
        beta_t = self.beta_schedule[t_unsq].sqrt()
        alpha_t = self.alpha_schedule[t_unsq].sqrt()
        
        x_prev_mean = (x - beta_t * noise_pred) / alpha_t

        alpha_bar_t_1 = self.alpha_bar_schedule[t_unsq - 1] if t.min() > 0 else self.alpha_bar_schedule[0]
        if t.min() == 0:  # Use the initial value if t contains 0
            alpha_bar_t_1[t == 0] = self.alpha_bar_schedule[0]
        sigma_t = (beta_t * (1 - alpha_bar_t_1) / (1 - alpha_t)).sqrt()

        # Sample x_(t-1)
        x_prev = x_prev_mean + sigma_t * torch.randn_like(x)
        return x_prev

    def p_sample_without_cell_type(self, x, t, noise=None):
        """
        Sample from the model's distribution p at time t without considering cell type.
        """
        if torch.all(t == 0):
            return x

        if noise is None:
            noise = torch.randn_like(x)

        # Compute the previous mean using the predicted noise
        # Since cell type is not considered, we'll assume noise_pred as random noise
        noise_pred = noise

        # Compute the variance of the reverse process
        t_unsq = t.unsqueeze(1)  
        beta_t = self.beta_schedule[t_unsq].sqrt()
        alpha_t = self.alpha_schedule[t_unsq].sqrt()
        
        x_prev_mean = (x - beta_t * noise_pred) / alpha_t

        alpha_bar_t_1 = self.alpha_bar_schedule[t_unsq - 1] if t.min() > 0 else self.alpha_bar_schedule[0]
        if t.min() == 0:  # Use the initial value if t contains 0
            alpha_bar_t_1[t == 0] = self.alpha_bar_schedule[0]
        sigma_t = (beta_t * (1 - alpha_bar_t_1) / (1 - alpha_t)).sqrt()

        # Sample x_(t-1)
        x_prev = x_prev_mean + sigma_t * noise
        return x_prev
   
    def interpolate_spatially(self, slice1_data, slice2_data, target_coords, t,lambda_val):
        interpolated_features = []

        # Convert list of tensors to a single tensor for computation
        features1 = torch.stack([data[0] for data in slice1_data])
        features2 = torch.stack([data[0] for data in slice2_data])
        coords1 = torch.stack([data[2] for data in slice1_data])
        coords2 = torch.stack([data[2] for data in slice2_data])

        # Normalize coordinates for distance calculation
        min_coords = torch.min(torch.cat([coords1, coords2, target_coords]), dim=0)[0]
        max_coords = torch.max(torch.cat([coords1, coords2, target_coords]), dim=0)[0]
        coords1_normalized = (coords1 - min_coords) / (max_coords - min_coords)
        coords2_normalized = (coords2 - min_coords) / (max_coords - min_coords)
        target_coords_normalized = (target_coords - min_coords) / (max_coords - min_coords)

        # Perform stochastic encoding for the gene expression features
        x1_t, _ = self.q_sample(features1, t)
        x2_t, _ = self.q_sample(features2, t)

        # Interpolate for each instance in the target slice
        for target_coord in target_coords_normalized:
            # Ensure the coordinates are 2D
            coords1_2d = coords1_normalized.unsqueeze(0) if coords1_normalized.ndim == 1 else coords1_normalized
            coords2_2d = coords2_normalized.unsqueeze(0) if coords2_normalized.ndim == 1 else coords2_normalized

            # Calculate distances
            distances1 = torch.norm(coords1_2d - target_coord, dim=1)
            distances2 = torch.norm(coords2_2d - target_coord, dim=1)
            closest_idx1 = torch.argmin(distances1)
            closest_idx2 = torch.argmin(distances2)

            # Get the features of the closest instances
            closest_feature1 = x1_t[closest_idx1]
            closest_feature2 = x2_t[closest_idx2]

            # Compute the interpolation factor based on proximity
            #lambda_val = distances1[closest_idx1] / (distances1[closest_idx1] + distances2[closest_idx2])

            # Perform the interpolation in latent space
            x_bar_t = (1 - lambda_val) * closest_feature1 + lambda_val * closest_feature2

            # Decode the interpolated latent representation
            t_tensor = torch.full((1,), t, dtype=torch.long, device=self.device)
            x_bar_0 = self.p_sample_without_cell_type(x_bar_t.unsqueeze(0), t_tensor)
            interpolated_features.append(x_bar_0.squeeze(0))

        # Concatenate all interpolated features into a tensor
        interpolated_features = torch.stack(interpolated_features)
        return interpolated_features

