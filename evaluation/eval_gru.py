from Models.gru_baseline import GRUBaseline
from evaluation.eval_runner import evaluate_model

if __name__ == '__main__':
    evaluate_model(model_name='gru', model_class=GRUBaseline, checkpoint_file='gru_fp32.pt')
