# -*- coding: utf-8 -*-

"""
Inference and submission script.
This script is used to make predictions on the test set and generate a submission file.
"""

import json
# from src.data_loader import get_test_loader
# from src.model import MyModel
# import torch

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    print("Configuration loaded:")
    print(config)

    # 1. Load test data
    # test_loader = get_test_loader(config)
    print("\nTest data loading placeholder...")

    # 2. Load trained model
    # model = MyModel(config)
    # checkpoint_path = 'outputs/checkpoints/best_model.pth'
    # model.load_state_dict(torch.load(checkpoint_path))
    # model.eval()
    print("Trained model loading placeholder...")

    # 3. Make predictions
    # predictions = []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         # outputs = model(batch)
    #         # _, predicted_labels = torch.max(outputs, 1)
    #         # predictions.extend(predicted_labels.tolist())
    #         pass
    print("Prediction generation placeholder...")

    # 4. Generate submission file
    # submission_df = pd.DataFrame({'id': test_loader.dataset.ids, 'answer': predictions})
    # submission_path = 'outputs/submissions/submission.csv'
    # submission_df.to_csv(submission_path, index=False)
    print("Submission file generation placeholder...")
    
    print(f"\nInference script finished. Submission file placeholder would be at 'outputs/submissions/submission.csv'")

if __name__ == '__main__':
    main()
