from tqdm import tqdm
import torch
import optree
from astra.torch.models import LearningToLoss
from torch.utils.data import TensorDataset, DataLoader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def LossPredLoss(input, target, margin=1.0):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    loss = torch.sum(torch.clamp(margin - one * input, min=0))
    loss = loss / input.size(0) # Note that the size of input is already halved
    return loss


def train_fn(model, inputs, outputs, loss_fn, lr, epochs, batch_size=None, shuffle=True, verbose=True, weight=1.0, margin=1.0):
    if batch_size is None:
        batch_size = len(inputs)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iter_losses = []
    epoch_losses = []

    # shuffle
    if shuffle:
        idx = torch.randperm(len(inputs))
    else:
        idx = torch.arange(len(inputs))

    for _ in range(epochs):
        loss_value = 0.0
        pbar = range(0, len(inputs), batch_size)
        if verbose:
            pbar = tqdm(pbar)
        for i in pbar:
            optimizer.zero_grad()
            if isinstance(model, LearningToLoss):
                pred, features = model.backbone(inputs[idx[i : i + batch_size]].to(model.device))
                target_loss = loss_fn(pred, outputs[idx[i : i + batch_size]].to(model.device))
                pred_loss = model.module(features)
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                m_module_loss = LossPredLoss(pred_loss, target_loss, margin=margin)
                loss = m_backbone_loss + weight * m_module_loss
            else:    
                pred = model(inputs[idx[i : i + batch_size]].to(model.device))
                loss = loss_fn(pred, outputs[idx[i : i + batch_size]].to(model.device))
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            loss_value += loss.item()
            if verbose:
                pbar.set_description(f"Loss: {loss.item():.6f}")

        # shuffle
        if shuffle:
            idx = torch.randperm(len(inputs))

        epoch_losses.append(loss_value / (len(inputs) / batch_size))
        if verbose:
            print(f"Epoch {len(epoch_losses)}: {epoch_losses[-1]}")

    return iter_losses, epoch_losses


def ravel_pytree(pytree):
    leaves, structure = optree.tree_flatten(pytree)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.numel() for leaf in leaves]
    flat_params = torch.cat([leaf.flatten() for leaf in leaves])

    def unravel_function(flat_params):
        assert flat_params.numel() == sum(sizes), f"Invalid flat_params size {flat_params.numel()} != {sum(sizes)}"
        assert len(flat_params.shape) == 1, f"Invalid flat_params shape {flat_params.shape}"
        flat_leaves = flat_params.split(sizes)
        leaves = [leaf.reshape(shape) for leaf, shape in zip(flat_leaves, shapes)]
        return optree.tree_unflatten(structure, leaves)

    return flat_params, unravel_function
