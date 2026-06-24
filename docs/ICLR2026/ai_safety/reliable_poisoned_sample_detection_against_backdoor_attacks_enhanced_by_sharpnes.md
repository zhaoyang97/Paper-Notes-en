---
title: >-
  [Paper Note] Reliable Poisoned Sample Detection against Backdoor Attacks Enhanced by Sharpness-Aware Minimization
description: >-
  [ICLR 2026][AI Safety][Sharpness-Aware Minimization] Academic paper note for Reliable Poisoned Sample Detection against Backdoor Attacks Enhanced by Sharpness-Aware Minimization.
tags:
  - ICLR 2026
  - AI Safety
  - Sharpness-Aware Minimization
date: 2026-05-08
content_hash: 0a2b4002022a6d36
---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 1. Defining the Trigger and Poisoning Logic (Mechanisms)
def apply_trigger(images, labels, trigger_value=1.0, poison_rate=0.05, target_label=0):
    """
    Simulates a simple backdoor attack by adding a trigger to a pixel.
    """
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    num_poisoned = int(len(images) * poison_rate)
    
    indices = torch.randperm(len(images))[:num_poisoned]
    # Simple trigger: set pixel (0,0) to trigger_value
    poisoned_images[indices, :, 0, 0] = trigger_value
    poisoned_labels[indices] = target_label
    
    is_poisoned = torch.zeros(len(images), dtype=torch.bool)
    is_poisoned[indices] = True
    
    return poisoned_images, poisoned_labels, is_poisoned

# 2. Sharpness-Aware Minimization (SAM) Implementation (Key Design)
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w"
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# 3. Method: Poisoned Sample Detection (Simplified Activation Clustering)
def detect_poisoned_samples(model, dataloader, device):
    """
    Simplified version of PSD: Uses internal activations to find outliers.
    In the paper, SAM is shown to sharpen the difference in these activations.
    """
    model.eval()
    activations = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # In a real scenario, hook the last conv layer. 
            # Here we use the penultimate layer output.
            feat = model.extract_features(inputs) 
            activations.append(feat.cpu())
    
    activations = torch.cat(activations, dim=0).numpy()
    
    # Simple PSD Logic: 
    # Use PCA to reduce dimensionality and then calculate scores (e.g., Spectral Signature)
    # The paper proves SAM increases Top-k TAC, leading to better separability here.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    reduced_feats = pca.fit_transform(activations)
    
    # Score based on the distance from the mean in the principal component space
    mean_feat = np.mean(reduced_feats, axis=0)
    scores = np.linalg.norm(reduced_feats - mean_feat, axis=1)
    
    return scores

# 4. Simple Model Architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(16 * 26 * 26, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x) # Features before last ReLU
        return x

# 5. Main Logic Simulation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy Dataset
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    
    # Poison the data
    X_poison, y_poison, mask = apply_trigger(X, y)
    
    dataset = torch.utils.data.TensorDataset(X_poison, y_poison)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleNet().to(device)
    
    # Use SAM to amplify backdoor effect
    base_opt = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_opt, rho=0.05, lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training Loop (Stage-1)
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # SAM First Step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM Second Step
        criterion(model(data), target).backward()
        optimizer.second_step(zero_grad=True)

    # Detection (Stage-2 & 3)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    poison_scores = detect_poisoned_samples(model, eval_loader, device)
    
    print(f"Average score for poisoned: {poison_scores[mask].mean():.4f}")
    print(f"Average score for clean: {poison_scores[~mask].mean():.4f}")
