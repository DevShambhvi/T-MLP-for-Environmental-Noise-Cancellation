import os
from Models.mlp import BasicMLP
from Models.t_mlp import TMLP
from Models.t_mlp_qat import TMLP_QAT
from Models.gru_baseline import GRUBaseline
from evaluation.eval_runner import evaluate_model

def run_all_evaluations():
    configs = [
        ("mlp", BasicMLP, "mlp_fp32.pt", False),
        ("t_mlp", TMLP, "t_mlp_fp32.pt", False),
        ("t_mlp_qat", TMLP_QAT, "t_mlp_int8.pt", True),
        ("gru", GRUBaseline, "gru_fp32.pt", False),
    ]

    all_results = []
    for name, model_cls, ckpt, is_int8 in configs:
        res = evaluate_model(name, model_cls, ckpt, is_int8)
        all_results.append(res)

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    report_path = os.path.join(root, "evaluation_results.md")

    # Generate evaluation_results.md
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Model Performance & Efficiency Comparison\n\n")
        f.write("| Model | Checkpoint | NMSE (dB) ↓ | Linear NMSE ↓ | F1-Score ↑ | Precision ↑ | Recall ↑ | Params | MACs/s | RAM Footprint |\n")
        f.write("|-------|------------|-------------|---------------|------------|-------------|----------|--------|--------|--------------|\n")

        for r in all_results:
            f.write(f"| **{r['model_name'].upper()}** | `{r['checkpoint']}` | **{r['nmse_db']:.2f} dB** | {r['nmse_lin']:.4f} | {r['f1_score']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['params']:,} | {r['macs_per_sec']:,} | {r['total_ram_kb']:.2f} KB |\n")

        f.write("\n\n## Quantization Analysis (FP32 vs INT8 T-MLP)\n\n")
        fp32_res = next(r for r in all_results if r['model_name'] == 't_mlp')
        int8_res = next(r for r in all_results if r['model_name'] == 't_mlp_qat')
        
        nmse_diff = int8_res['nmse_db'] - fp32_res['nmse_db']
        ram_saving = ((fp32_res['total_ram_kb'] - int8_res['total_ram_kb']) / fp32_res['total_ram_kb']) * 100.0

        f.write(f"- **FP32 T-MLP NMSE**: `{fp32_res['nmse_db']:.2f} dB`\n")
        f.write(f"- **INT8 T-MLP NMSE**: `{int8_res['nmse_db']:.2f} dB` (Δ = `{nmse_diff:+.2f} dB`)\n")
        f.write(f"- **Memory Savings**: FP32 `{fp32_res['total_ram_kb']:.2f} KB` → INT8 `{int8_res['total_ram_kb']:.2f} KB` (**~{ram_saving:.1f}% RAM reduction**)\n")

    print(f"\n[OK] All evaluations complete! Report updated: {report_path}")

if __name__ == '__main__':
    run_all_evaluations()
