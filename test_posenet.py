import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# ----------------------------
# 1. Enhanced Dataset with Better Augmentations (Bridge Sim2Real Gap)
# ----------------------------
class EnhancedPoseDataset(Dataset):
    """
    This class prepares the dataset for training the 3D pose estimation model, with a special focus on bridging the Sim2Real gap. 
    It reads data, applies robust augmentations to synthetic images, 
    and converts the data into a format (PyTorch Tensors) suitable for a PyTorch model.

    Data augmentation: Modifying the synthetic data via diff variations (lighting/weather/hue), 
    for the model to train and identify features of the place, so it could generalise better. 
    """

    def __init__(self, txt_file, img_dir, augment=False, is_synthetic=False):
        self.img_dir = img_dir
        self.augment = augment
        self.is_synthetic = is_synthetic
        self.samples = []

        with open(txt_file, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4 or "NaN" in parts: # skip invalid lines
                    continue
                fname, x, y, z = parts
                self.samples.append((fname, [float(x), float(y), float(z)]))

        # Augmentations based on research papers
        if augment and is_synthetic:
            # Stronger augmentations for synthetic data to bridge domain gap
            self.transform = A.Compose([ # Compose -- chains tgt a series of augmentations
                A.Resize(299, 299),  # InceptionV3 input size
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type="drizzle", p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=3, fill_value=0, p=0.3),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                # ImageNet normalization -- normalize image using the mean and std dev of the ImageNet dataset
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(), # Convert image to PyTorch tensor -- numpy array
            ])
        
        else:
            # Less preprocessing for test/non-synthetic data
            self.transform = A.Compose([
                A.Resize(299, 299),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        # return dataset length
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve filename and corresponding 3D coords
        fname, xyz = self.samples[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(fname))
        img = Image.open(img_path).convert("RGB") # Open image and convert to RGB
        img_array = np.array(img)
        
        # Apply augmentations
        # note: shape of img_array: (H, W, C) (299, 299, 3), output (C, H, W)
        # transformed is a dictionary of transformed inputs e.g. img_array
        transformed = self.transform(image=img_array)

        img_tensor = transformed["image"] # ToTensorV2() will hold 'image' data as PyTorch tensor
        
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32) # converts 3D coords into a PyTorch tensor
        return img_tensor, xyz_tensor

# ----------------------------
# 2. InceptionV3-based Architecture (Following Research Papers)
# ----------------------------
class InceptionV3PoseNet(nn.Module):
    """
    This model uses InceptionV3 as a feature extractor, then passes the features
    through a custom regressor to predict 3D coordinates (x, y, z) for pose estimation.
    """
    
    def __init__(self, pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        # Use InceptionV3 as feature extractor (following paper 1)
        self.encoder = timm.create_model(
            'inception_v3', 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head, network will output a feature vector (2048 numbers)
            
            # Global max pooling -- reduce a 3D feature map (C x H x W) to a 1D vector of length C
            # by taking the maximum value across each channel’s H x W map
            global_pool='max'
        )
        
        # Get the feature dimension (2048 for InceptionV3)
        feature_dim = self.encoder.num_features
        
        # Regressor following the papers' architecture: 2048 -> 4096 -> 4096 -> 3
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate), # prevent overfitting
            nn.Linear(feature_dim, 4096), # makes weighted sum
            nn.ReLU(), # non linear activation
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3) # output layer
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.regressor.modules(): # self.regressor.modules() returns all layers inside self.regressor sequentially 
            if isinstance(m, nn.Linear):
                '''
                xavier_uniform: sets the weights to values that keep variance of gradients roughly the same across layers
                
                weight = weight - learning_rate * gradient
                - weights too big: gradients explode --> training diverges
                - weights too small -> gradients vanish --> training too slow, stalls
                '''
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: # init biases to 0
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        '''
        Forward pass of the model:
        1. Extract features from the input image
        2. Pass features through the regressor to get 3D coords
        '''
        # Extract features using InceptionV3
        features = self.encoder(x)  # (batch_size, 2048)
        
        # Regress to 3D coordinates
        coords = self.regressor(features)  # (batch_size, 3)
        return coords

# ----------------------------
# 3. Domain Adaptation Loss (Optional - for advanced Sim2Real bridging)
# ----------------------------

class DomainAdversarialLoss(nn.Module):
    """
    Domain Adversarial Loss for bridging the gap between synthetic and real data.

    Purpose:
    This loss function helps the main network learn to extract features that look similar
    for both synthetic (simulated) and real-world images. This allows the model
    to generalize better to real-world data, even when primarily trained on synthetic data.

    How it Works (The Student-Teacher Game):
    - A small domain classifier (the "Teacher") is trained to correctly identify if a feature comes from a synthetic or real image.
    - The main network (the "Student") is trained to predict the correct pose, but also to intentionally confuse the Teacher.
    - This is achieved via gradient reversal, where the main network's total loss is calculated as `pose_loss - 0.1 * domain_loss`. 
    By minimizing this total loss, the main network is forced to both improve its pose prediction and increase the domain classifier's loss, thus making the features more similar.
    """

    def __init__(self, feature_dim=2048, hidden_dim=1024):
        '''
        - feature_dim (int): The dimensionality of the input features from the main network's encoder.
        - hidden_dim (int): The number of neurons in the hidden layer of the domain classifier network.
        '''
        super().__init__() # initialize the parent nn.Module class
        self.domain_classifier = nn.Sequential(
            # define a small neural network (domain classifier) to predict the data type (synthetic/real) 
            nn.Linear(feature_dim, hidden_dim), # fully connected layer: input features -> hidden layer
            nn.ReLU(), # non-linear activation function
            nn.Dropout(0.5), # randomly turn off 50% of neurons during training to prevent overfitting
            nn.Linear(hidden_dim, 1), # final layer maps hidden layer to single output (probability of being real)
            nn.Sigmoid() # squashes output to range [0, 1], suitable for probability
        )
    

    def forward(self, features, domain_labels):
        '''
        Computes the domain classification loss.

        Args:
            features: Tensor of shape (batch_size, feature_dim), extracted from the encoder.
            domain_labels: Tensor of shape (batch_size,), with 0s for synthetic features and 1s for real features.

        Returns:
            loss: A scalar BCELoss indicating how well the classifier predicts the domain.
        '''

        # pass the extracted features through the domain classifier network
        domain_pred = self.domain_classifier(features)
        # Compute binary cross-entropy loss:
        # domain_pred.squeeze() removes extra dimension if shape is (batch_size, 1) -> (batch_size,)
        # domain_labels.float() converts labels from int to float (BCELoss expects float)
        loss = nn.BCELoss()(domain_pred.squeeze(), domain_labels.float())
        return loss # return the loss value to use in training

# ----------------------------
# 4. Enhanced Training Function with Multiple Loss Components
# ----------------------------
def train_enhanced_model(model, train_loader_synthetic, train_loader_real, 
                        val_loader, device, epochs, use_domain_adaptation=False):
    """
    Train the pose estimation model using synthetic (and optionally real) images.
    Optionally includes domain adaptation to bridge the gap between synthetic and real data.

    Args:
        model: The 3D pose estimation neural network.
        train_loader_synthetic: Data loader for synthetic training images.
        train_loader_real: Data loader for real training images (optional).
        val_loader: Data loader for validation/testing images.
        device: 'cuda' or 'cpu' for computation.
        epochs: Number of training cycles over the dataset.
        use_domain_adaptation: Whether to include domain adversarial loss.
    """

    # Optimizer
    '''
    - model.parameters(): collection of all learnable weights and biases in the neural network
    - lr, learning rate: controls the step size the optimizer takes in the direction of the min. loss
    - betas(mometum, adaptive LR): Beta_1 smoothes out the optimization path, Beta_2 gives each weight its own, dynamic learning rate
    - eps: epsilon, tiny constant added to the denominator to prevent div by 0
    '''
    optimizer = optim.NAdam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    # Scheduler - reduce Learning rate when validation error stops improving
    '''
    - optimizer: wrapped optimizer
    - mode: lr will be reduced when the qty monitored has stopped decreasing, vv for max
    - factor: factor by which the lr will be reduced
    - patience: number of allowed epochs with no improvement after which the learning rate will be reduced
    '''
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    # Loss functions: mean squared error betw predicted and true 3D coords
    pose_criterion = nn.MSELoss()  # MSE as in paper 1, or try MAE as in paper 2
    
    # Optional domain adaptation: helps model generalise from synthetic to real images
    domain_loss_fn = DomainAdversarialLoss().to(device) if use_domain_adaptation else None
    domain_optimizer = optim.Adam(domain_loss_fn.parameters(), lr=0.001) if use_domain_adaptation else None
    
    best_val_error = float('inf') # keep track of best validation performance
    
    for epoch in range(epochs):
        model.train() # training mode
        total_pose_loss = 0
        total_domain_loss = 0
        num_batches = 0
        
        # Iterators to go through synthetic and real data
        synthetic_iter = iter(train_loader_synthetic)
        real_iter = iter(train_loader_real) if train_loader_real else None
        
        for batch_idx, (syn_imgs, syn_coords) in enumerate(synthetic_iter):
            # move data to GPU if available
            syn_imgs, syn_coords = syn_imgs.to(device), syn_coords.to(device)
            
            # Get real batch if available
            real_imgs, real_coords = None, None
            if real_iter:
                try:
                    real_imgs, real_coords = next(real_iter)
                    real_imgs, real_coords = real_imgs.to(device), real_coords.to(device)
                except StopIteration:
                    # restart real iterator if we reach the end
                    real_iter = iter(train_loader_real)
                    real_imgs, real_coords = next(real_iter)
                    real_imgs, real_coords = real_imgs.to(device), real_coords.to(device)
            
            optimizer.zero_grad() # reset gradients
            
            # Forward pass on synthetic data: Predict 3D coords for synthetic images
            syn_pred = model(syn_imgs)
            pose_loss = pose_criterion(syn_pred, syn_coords) # compute pose loss -- MSE
            
            # Include real data loss if available
            if real_imgs is not None:
                real_pred = model(real_imgs)
                pose_loss += pose_criterion(real_pred, real_coords)
            
            # Domain adaptation loss (optional)
            domain_loss = 0
            if use_domain_adaptation and real_imgs is not None:
                # Extract features for domain classification (without updating the encoder)
                with torch.no_grad(): # domain classifier is trained on features from frozen inceptionV3 feature extractor (before intended gradient reversal step occurs)
                    syn_features = model.encoder(syn_imgs) # weights of InceptionV3 are fixed, NOT updated
                    real_features = model.encoder(real_imgs)
                
                # Create domain labels (0 for synthetic, 1 for real)
                syn_domain_labels = torch.zeros(syn_imgs.size(0)).to(device)
                real_domain_labels = torch.ones(real_imgs.size(0)).to(device)
                
                # Compute domain classification loss
                domain_loss = domain_loss_fn(syn_features, syn_domain_labels) + \
                             domain_loss_fn(real_features, real_domain_labels)
                
                # Update domain classifier to improve its predictions
                domain_optimizer.zero_grad()
                domain_loss.backward(retain_graph=True) # calc gradients for the domain classifier only (minimizing domain_loss)
                domain_optimizer.step() # update weights of domain classifier only (using its dedicated Adam optimizer)
            
            # Backpropagation for pose estimation
            total_loss = pose_loss - 0.1 * domain_loss  # Total loss: pose loss - small factor of domain loss (gradient reversal)
            total_loss.backward() # main network weights are updated
            
            # Gradient clipping
            # prevent large gradient updates, max gradient cap
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # update model weights

            # Accumulate losses for reporting            
            total_pose_loss += pose_loss.item()
            total_domain_loss += domain_loss if isinstance(domain_loss, (int, float)) else domain_loss.item()
            num_batches += 1
        
        # Evaluate model on validation set
        val_error = evaluate_model(model, val_loader, device)
        scheduler.step(val_error) # adjust learning rate if validation stops improving
        
        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error

            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "models")
            os.makedirs("models", exist_ok = True)
            
            torch.save(model.state_dict(), 'models/best_pose_model.pth')
        
        avg_pose_loss = total_pose_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Pose Loss={avg_pose_loss:.4f}, "
              f"Domain Loss={avg_domain_loss:.4f}, "
              f"Val Error={val_error:.4f}m, "
              f"Best={best_val_error:.4f}m")
    
    return best_val_error

# evaluate model
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    '''
    Compute median distance error between predicted and true 3D positions.
    '''
    model.eval() # set to evaluation mode
    errors = []
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        dists = torch.norm(preds - targets, dim=1) # output the Euclidean distance
        errors.extend(dists.cpu().numpy())
    return np.median(errors)

# visualise predictions
@torch.no_grad() # do not compute gradients
def visualize_predictions(model, dataloader, device, save_path):
    """
    Visualize predicted 3D trajectory vs ground truth
    """
    model.eval() # set to evaluation mode
    all_preds = []
    all_targets = []

    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Combine all batches
    all_preds = np.concatenate(all_preds, axis=0)  # shape: (N, 3)
    all_targets = np.concatenate(all_targets, axis=0)  # shape: (N, 3)

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.plot(all_targets[:, 0], all_targets[:, 1], all_targets[:, 2], 
            label='Ground Truth', color='red', marker='o')
    
    # Plot predicted path
    ax.plot(all_preds[:, 0], all_preds[:, 1], all_preds[:, 2], 
            label='Predicted', color='blue', marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Predicted vs Ground Truth 3D Path')
    ax.legend()
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig(save_path)
    print(f"Saved 3D path visualization to: {save_path}")
    plt.close()

# ----------------------------
# 5. Main Training Pipeline
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths (update these to your paths)
    train_txt = r"C:\Users\Zihui\Downloads\SUTD_T3\AIR_Lab\lab-3d\dataset_train_unity.txt"
    train_img_dir = r"C:\Users\Zihui\Downloads\SUTD_T3\AIR_Lab\lab-3d\synthetic3"
    test_txt = r"C:\Users\Zihui\Downloads\SUTD_T3\AIR_Lab\lab-3d\dataset_vid.txt"
    test_img_dir = r"C:\Users\Zihui\Downloads\SUTD_T3\AIR_Lab\lab-3d\rtk2"
    
    # Create datasets with enhanced augmentations
    train_synthetic_dataset = EnhancedPoseDataset(
        train_txt, train_img_dir, augment=True, is_synthetic=True
    )
    
    
    test_dataset = EnhancedPoseDataset(
        test_txt, test_img_dir, augment=False, is_synthetic=False
    )
    
    # Create data loaders
    # training data loader
    train_synthetic_loader = DataLoader(
        train_synthetic_dataset, batch_size=24, shuffle=True, num_workers=4
    )
    
    # Since I dun have real training data, set to None
    train_real_loader = None
    
    # testing data loader
    val_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    
    # Create enhanced model
    model = InceptionV3PoseNet(pretrained=True, dropout_rate=0.5).to(device)
    
    # Train the model
    print("Starting enhanced training with InceptionV3...")
    best_error = train_enhanced_model(
        model, train_synthetic_loader, train_real_loader, val_loader, 
        device, epochs=20, use_domain_adaptation=False
    )
    
    print(f"Best validation error: {best_error:.4f}m")

    # Visualize predictions on test set
    save_path = r"C:\Users\Zihui\Downloads\SUTD_T3\AIR_Lab\cv_training_v0\pose_prediction_analysis.png"
    visualize_predictions(model, val_loader, device, save_path)

if __name__ == "__main__":
    main()