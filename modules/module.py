import torch

def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q, epsilon=1e-6):
    # Add a small value epsilon to avoid division by zero
    weight = q**2 / (q.sum(0) + epsilon)
    
    # Ensure no division by zero during transpose
    transposed_weight = weight.t()
    normalized_weight = transposed_weight / (transposed_weight.sum(1, keepdim=True) + epsilon)
    
    return normalized_weight.t()

