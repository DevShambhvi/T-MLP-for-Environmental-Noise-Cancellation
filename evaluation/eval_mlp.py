from Models.mlp import BasicMLP
from evaluation.eval_runner import evaluate_model

if __name__ == '__main__':
    evaluate_model(model_name='mlp', model_class=BasicMLP, checkpoint_file='mlp_fp32.pt')
