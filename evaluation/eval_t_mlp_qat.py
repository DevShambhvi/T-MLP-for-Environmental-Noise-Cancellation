from Models.t_mlp_qat import TMLP_QAT
from evaluation.eval_runner import evaluate_model

if __name__ == '__main__':
    evaluate_model(model_name='t_mlp_qat', model_class=TMLP_QAT, checkpoint_file='t_mlp_int8.pt', is_int8=True)
