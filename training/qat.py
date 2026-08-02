import os, argparse, torch, torch.nn as nn
from DataPipeline.dataset import create_dataloaders
from Models.t_mlp import TMLP
from Models.t_mlp_qat import TMLP_QAT

def main():
    p = argparse.ArgumentParser(description="QAT fine-tuning for T-MLP")
    p.add_argument('--checkpoint', default=None,
                   help="Path to FP32 T-MLP checkpoint (default: results/t_mlp_fp32.pt)")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    args = p.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ckpt = args.checkpoint or os.path.join(root, 'results', 't_mlp_fp32.pt')

    # Load FP32 weights into QAT model
    fp32 = TMLP()
    fp32.load_state_dict(torch.load(ckpt, weights_only=True))
    qat = TMLP_QAT()
    qat.net.load_state_dict(fp32.net.state_dict())
    qat.fuse_and_prepare()
    print(f"Loaded FP32 checkpoint: {ckpt}")

    # Data
    train_ld, val_ld, _ = create_dataloaders(
        os.path.join(root, 'data', 'speech'),
        os.path.join(root, 'data', 'noise'), batch_size=args.batch,
    )

    opt = torch.optim.Adam(qat.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for e in range(1, args.epochs + 1):
        qat.train()
        total, n = 0, 0
        for x, y in train_ld:
            opt.zero_grad()
            loss = loss_fn(qat(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"QAT Epoch {e:2d}/{args.epochs} | Loss: {total/n:.6f}")

    # Convert & save
    qat.eval()
    qat.convert_to_int8()
    save_path = os.path.join(root, 'results', 't_mlp_int8.pt')
    torch.save(qat.state_dict(), save_path)
    print(f"INT8 model saved: {save_path}")

if __name__ == '__main__':
    main()
