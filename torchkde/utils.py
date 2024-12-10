def ensure_two_dimensional(tensor):
    if tensor.dim() == 0:  # Scalar tensor
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Make it 1x1
    elif tensor.dim() == 1:  # 1D tensor
        tensor = tensor.unsqueeze(0)  # Add a batch dimension
    return tensor
