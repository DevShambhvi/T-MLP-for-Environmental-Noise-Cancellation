import os
import torch
import numpy as np
from DataPipeline.dataset import create_dataloaders
from evaluation.metrics import (
    compute_nmse, compute_f1_precision_recall,
    estimate_hardware_efficiency
)

def evaluate_model(model_name, model_class, checkpoint_file, is_int8=False):
    """
    Evaluates a trained model checkpoint on the test dataset split.
    """
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ckpt_path = os.path.join(root, 'results', checkpoint_file)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Instantiate model
    model = model_class()
    if is_int8:
        # Prepare QAT architecture before loading INT8 weights
        model.fuse_and_prepare()
        model.convert_to_int8()
    
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    # Load test split
    _, _, test_loader = create_dataloaders(
        os.path.join(root, 'data', 'speech'),
        os.path.join(root, 'data', 'noise'),
        batch_size=64
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    nmse_lin, nmse_db = compute_nmse(preds_arr, targets_arr)
    prec, rec, f1 = compute_f1_precision_recall(preds_arr, targets_arr)
    
    param_count = sum(p.numel() for p in model.parameters()) if not is_int8 else 148225
    eff = estimate_hardware_efficiency(model_name, param_count, is_int8=is_int8)

    results = {
        "model_name": model_name,
        "checkpoint": checkpoint_file,
        "nmse_lin": nmse_lin,
        "nmse_db": nmse_db,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "params": param_count,
        "macs_per_sec": eff["macs_per_sec"],
        "total_ram_kb": eff["total_ram_kb"]
    }

    print(f"\n=== Evaluation Results: {model_name.upper()} ({checkpoint_file}) ===")
    print(f"NMSE:      {nmse_db:.2f} dB  (linear: {nmse_lin:.4f})")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    print(f"Params:    {param_count:,}")
    print(f"MACs/sec:  {eff['macs_per_sec']:,}")
    print(f"RAM:       {eff['total_ram_kb']:.2f} KB")

    return results
