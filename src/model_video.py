"""
src/model_video.py
Temporal Video Model for PRISM.
Uses a 3D ResNet backbone to analyze spatial-temporal motion anomalies across frame sequences.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models

class PRISMVideoNetwork(nn.Module):
    """
    3D Convolutional Neural Network for Video Deepfake Detection.
    Captures temporal flickering, inter-frame inconsistencies, and motion jitter.
    
    Expected Input: [Batch, Channels (3), Frames (T), Height (224), Width (224)]
    """
    def __init__(self, pretrained: bool = True, fine_tune_layers: int = 2) -> None:
        super().__init__()
        
        # Load pre-trained R3D_18 (ResNet3D 18-layer) trained on Kinetics-400
        # It's an excellent feature extractor for spatio-temporal video data.
        weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = video_models.r3d_18(weights=weights)
        
        # Freeze early layers to save VRAM and avoid overfitting
        # Only fine-tune the last `fine_tune_layers` blocks + the fully connected layer
        blocks = list(self.backbone.children())
        for i, block in enumerate(blocks):
            if i < len(blocks) - fine_tune_layers - 1:
                for param in block.parameters():
                    param.requires_grad = False
                    
        # Replace the final classification head (400 classes -> 1 binary class)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> dict:
        # x shape: [B, 3, T, 224, 224]
        logit = self.backbone(x)
        return {"logit": logit}

if __name__ == "__main__":
    # Smoke test for VRAM and shape validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PRISMVideoNetwork on {device}...")
    
    model = PRISMVideoNetwork(pretrained=False).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {train_params:,}")
    
    # Batch = 2, Channels = 3, Frames = 16, H=224, W=224
    dummy_video = torch.randn(2, 3, 16, 224, 224).to(device)
    
    with torch.no_grad():
        out = model(dummy_video)
        
    print(f"Logit Shape: {out['logit'].shape}")
    assert out["logit"].shape == (2, 1), "Output shape mismatch!"
    print("Smoke test passed successfully!")
