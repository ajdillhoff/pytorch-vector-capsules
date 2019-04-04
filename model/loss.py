import torch.nn.functional as F


def margin_loss(x, labels):
    m_plus = 0.9
    m_minus = 0.1
    lam = 0.5
    left = F.relu(m_plus - x).view(x.shape[0], -1)
    right = F.relu(x - m_minus).view(x.shape[0], -1)
    loss = labels * left + lam * (1.0 - labels) * right
    loss = loss.sum(dim=1)
    return loss.mean()


def reconstruction_loss(x, img):
    """Computes the loss between the reconstructed image and the true image.

    Args:
        x (N,1,28,28): Reconstructed image.
        img (N,1,28,28): True image.
    Returns:
        MSE loss.
    """
    return F.mse_loss(x.view(x.shape[0], -1), img.view(img.shape[0], -1))
