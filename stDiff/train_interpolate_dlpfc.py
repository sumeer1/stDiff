import argparse
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import Compose, Lambda
from dataset import SpatialTranscriptomicDataset
from model import DiffusionModel, MyDDPM

def parse_args():
    parser = argparse.ArgumentParser(description="Train a spatial transcriptomic diffusion model")
    parser.add_argument('--data-file', type=str, required=True, help="Path to the CSV dataset")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs to train for")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument('--output-file', type=str, required=True, help="File path to save the model and other outputs")
    parser.add_argument('--exclude-bregma', type=int, default=4, help="Bregma value to exclude from the dataset")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    transform = Compose([
        Lambda(lambda x: (x - torch.mean(x)) / torch.std(x))
    ])
    
    dataframe = pd.read_csv(args.data_file)
    num_samples = len(dataframe)
    num_train_samples = int(num_samples * 0.8)
    train_dataframe = dataframe[:num_train_samples]
    val_dataframe = dataframe[num_train_samples:]
    
    train_dataset = SpatialTranscriptomicDataset(train_dataframe, exclude_bregma=args.exclude_bregma, transform=transform)
    val_dataset = SpatialTranscriptomicDataset(val_dataframe, exclude_bregma=args.exclude_bregma, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #n_epochs = 300  # Example number of epochs
    #lr = 1e-3  # Learning rate
    patience = 10 # For early stopping
    best_val_loss = float('inf')
    num_epochs_no_improvement = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_genes = train_dataset.features.shape[1]
    network = DiffusionModel(num_genes=num_genes)
    ddpm = MyDDPM(network=network, n_steps=1000, device=device, num_genes=num_genes)
    ddpm.to(device)
    
    optimizer = Adam(ddpm.parameters(), lr=args.learning_rate)
    scheduler = OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum=False)

    base_dir, model_filename = os.path.split(args.output_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    
        # Training loop
    for epoch in range(args.epochs):
        ddpm.train()
        epoch_loss = 0.0
        for step, (x0, spatial_coords, slice_index) in enumerate(train_loader):
            x0 = x0.to(device)
            #cell_types = cell_types.to(device)
            #t = torch.randint(0, ddpm.n_steps, (len(x0),)).to(device)
            t = torch.randint(0, ddpm.n_steps, (x0.size(0),)).to(device)
            optimizer.zero_grad()
    
            # Compute the losses for the p distribution (reversed process)
            loss = ddpm.p_losses(x0, t)
            epoch_loss += loss.item()
    
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss / len(train_loader)}")
    
        # Validation loop
        ddpm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, (x0_val, spatial_coords_val, slice_index_val) in enumerate(val_loader):
                x0_val = x0_val.to(device)
                #val_cell_types = val_cell_types.to(device)
                t_val = torch.randint(0, ddpm.n_steps, (len(x0_val),)).to(device)
                val_loss += ddpm.p_losses(x0_val, t_val).item()
            
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")
    
        #scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_no_improvement = 0
            # Save the model when validation loss decreases
            #torch.save(ddpm.state_dict(), 'ddpm_model_lambda05.pt')
        else:
            num_epochs_no_improvement += 1
    
        if num_epochs_no_improvement >= patience:
            print('Early stopping')
            break  # Break the training loop
    
        # After the validation loop
        with torch.no_grad():
            # Define specific Bregma values for interpolation
            bregma_value1 = 3    
            bregma_value2 = 5  
            target_bregma = 4    
            
            full_dataset1 = SpatialTranscriptomicDataset(dataframe, transform=transform)
            # Retrieve the data for the slices
            slice1_data = train_dataset.select_slices(bregma_value1)
            slice2_data = train_dataset.select_slices(bregma_value2)
            target_slice_data = full_dataset1.select_slices(target_bregma)
    
            if slice1_data and slice2_data and target_slice_data:
                # Ensure all data is on the same device as the model
                slice1_data = [(data[0].to(device), data[1], data[2].to(device)) for data in slice1_data]
                slice2_data = [(data[0].to(device), data[1], data[2].to(device)) for data in slice2_data]
                target_slice_data = [(data[0].to(device), data[1], data[2].to(device)) for data in target_slice_data]
    
                # Obtain the spatial coordinates for the target slice and move them to the same device
                target_coords = torch.stack([data[2] for data in target_slice_data]).to(device)
    
    
                for lambda_val in np.arange(0.1, 1.0, 0.1):
                    interpolated_profiles = ddpm.interpolate_spatially(
                        slice1_data, slice2_data, target_coords, t=100, lambda_val=lambda_val
                    )
    
                    # Convert to numpy array and save to CSV
                    
                    interpolated_data = interpolated_profiles.cpu().numpy()
                    #interpolated_data  = np.log1p(interpolated_profiles.cpu().numpy())
                    interpolated_save_path = os.path.join(base_dir, f"dlpfclayer4_lambda_{lambda_val:.1f}.csv")
                    pd.DataFrame(interpolated_data).to_csv(interpolated_save_path, index=False)
                    
    
                
            else:
                print("Required slices for interpolation not found.")
    
if __name__ == '__main__':
    main()

