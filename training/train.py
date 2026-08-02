import os, argparse, torch, torch.nn as nn
from DataPipeline.dataset import create_dataloaders
from Models.mlp import BasicMLP
from Models.t_mlp import TMLP
from Models.gru_baseline import GRUBaseline

MODELS = {'mlp': BasicMLP, 't_mlp': TMLP, 'gru': GRUBaseline}

def train_epoch(model, loader, opt, loss_fn, dev):
    model.train()
    total, n = 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
        n += 1
    return total / n

@torch.no_grad()
def val_epoch(model, loader, loss_fn, dev):
    model.eval()
    total, n = 0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        total += loss_fn(model(x), y).item()
        n += 1
    return total / n

def main():
    p = argparse.ArgumentParser(description="Train FP32 models")
    p.add_argument('--model', choices=MODELS, default='t_mlp')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10)
    args = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    train_ld, val_ld, _ = create_dataloaders(
        os.path.join(root, 'data', 'speech'),
        os.path.join(root, 'data', 'noise'), batch_size=args.batch,
    )

    model = MODELS[args.model]().to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Params: {params:,} | Device: {dev}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val, wait = float('inf'), 0
    os.makedirs(os.path.join(root, 'results'), exist_ok=True)
    save_path = os.path.join(root, 'results', f'{args.model}_fp32.pt')

    for e in range(1, args.epochs + 1):
        t = train_epoch(model, train_ld, opt, loss_fn, dev)
        v = val_epoch(model, val_ld, loss_fn, dev)
        tag = ''
        if v < best_val:
            best_val, wait = v, 0
            torch.save(model.state_dict(), save_path)
            tag = ' [saved]'
        else:
            wait += 1
        print(f"Epoch {e:3d}/{args.epochs} | Train: {t:.6f} | Val: {v:.6f}{tag}")
        if wait >= args.patience:
            print(f"Early stopping at epoch {e}")
            break

    print(f"\nBest val loss: {best_val:.6f} -> {save_path}")

if __name__ == '__main__':
    main()
