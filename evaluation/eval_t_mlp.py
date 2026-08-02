from Models.t_mlp import TMLP
from evaluation.eval_runner import evaluate_model

if __name__ == '__main__':
    evaluate_model(model_name='t_mlp', model_class=TMLP, checkpoint_file='t_mlp_fp32.pt')
