import torch
import random
import numpy as np
import os
import sys
import pandas as pd
import ast
import time
import wandb 
from sklearn.metrics import f1_score, accuracy_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
  sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.agent.nodes.solver import SolverNode

class SolverEvaluation:
  def __init__(self, config):
    self.config = config
    self.model = SolverNode(config)
    self.data = []
    self.project_root = project_root
    self.__load_data()
    self.true_answers = []
    self.pred_answers = []

  def __call__(self, dataset):
    result = []
    for data in dataset:
      res = self.model(data)
      result.append(res)
      self.pred_answers.append(res['solver_results'][0]['answer'])
      self.true_answers.append(data['answer'])
    return result

  def __load_data(self):
    data_path = os.path.join(self.project_root, self.config.path.data.validate)
    df = pd.read_csv(data_path)

    for _, row in df.iterrows():
      problems_dict = ast.literal_eval(row['problems'])
      if row['rag'] != 'F':
        continue
      self.data.append({
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems_dict['question'],
        'question_plus': row.get('question_plus', None),
        'choices': problems_dict['choices'],
        'answer': row['answer'],
      })
    print(len(self.data), "length of data")

def set_random_seed(random_seed=42):
  random.seed(random_seed)
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

def add_project_root():
  project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
  if project_root not in sys.path:
    sys.path.append(project_root)


if __name__ == "__main__":
  set_random_seed()
  cfg = load_config()
  experiment_name = "no think, rag F, qwen3-32b"
  wandb.init(
    project="solver-evaluation",
    name=f"solver-eval-{experiment_name}"
  )

  solver = SolverEvaluation(cfg)

  start_time = time.time()
  result = solver(solver.data)
  elapsed_time = time.time() - start_time

  log_data = []

  for i, item in enumerate(result):
    for res in item['solver_results']:
      print(f" 풀이: {res['reasoning']}... → 원본답: {res['answer']}")

      log_data.append([
        solver.data[i]['id'],
        res['reasoning'],
        res['answer'],
        solver.true_answers[i]
      ])
  print(f"\n 추론 시간: {elapsed_time:.2f}초 (문제당 {elapsed_time/len(result):.2f}초)")
  
  f1 = f1_score(solver.true_answers, solver.pred_answers, average='macro')
  accuracy = accuracy_score(solver.true_answers, solver.pred_answers)
  print(f" F1 Score: {f1:.4f}")
  print(f" Accuracy: {accuracy:.4f}")
  print(log_data)
  wandb.log({
    "f1_score": f1,
    "accuracy": accuracy,
    "elapsed_time": elapsed_time,
    "evaluation_results": wandb.Table(
      data=log_data,
      columns=["id", "reasoning", "pred_answer", "true_answer"]
    )
  })
  wandb.finish()