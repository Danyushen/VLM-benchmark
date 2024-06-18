import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
import os
import os.path as osp
from tqdm import tqdm

base_dir = 'data/GQA'

# Set the seed
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

# Ensure reproducibility in CuDNN (for Convolutional operations), not sure if needed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = AutoModel.from_pretrained('weights/hf/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('weights/hf/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

# Load test-dev dataset
with open(osp.join(base_dir, 'questions/testdev_balanced_questions.json')) as f:
    data_questions = json.load(f)

# get the answer from the model for the given question and image
def get_answer(question, image_id):
    image_path = os.path.join(base_dir, 'images', image_id + '.jpg')
    image = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': question}]
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )
    return res

# get the predictions for all the questions in the dataset
def get_predictions(data_keys):
    data_keys = list(data_questions.keys())
    predictions_list = []
    image_ids_list = []
    answers_list = []
    questions_list = []

    for data in tqdm(data_keys):
        predictions = {}
        question_type = data_questions[data]['types']['structural']
        if question_type == 'verify':
            image_id = data_questions[data]['imageId']
            question = data_questions[data]['question']
            question_id = data
            instruction = 'Provide a short answer to the question and always start the answer with yes or no:'
            prediction = get_answer(instruction + question, image_id)

            # save predictions as [{"questionId": str, "prediction": str}]
            predictions['questionId'] = question_id
            predictions['prediction'] = prediction
            predictions_list.append(predictions)
            image_ids_list.append(image_id)
            answers_list.append(data_questions[data]['answer'])
            questions_list.append(question)

    return predictions_list, image_ids_list, answers_list, questions_list

# compute the accuracy
def evaluate(predictions_list, answers_list):
    # get the first word of each prediction
    predictions_short = [predictions_list[i]['prediction'].split(' ')[0] for i, _ in enumerate(predictions_list)]
    # lower case and remove punctuation (, and .) from the predictions
    predictions_short = [prediction.lower().replace(',', '').replace('.', '') for prediction in predictions_short]
    # check if the number of predictions and answers match
    if len(predictions_short) != len(answers_list):
        print('The number of predictions and answers do not match')
        return
    # calculate the accuracy
    else:
        idx_wrong = []
        correct = 0
        for i, _ in enumerate(predictions_short):
            if predictions_short[i] == answers_list[i]:
                correct += 1
            else:
                idx_wrong.append(i)
        accuracy = correct / len(predictions_short)
        return accuracy, idx_wrong



if __name__ == '__main__':
    predictions_list, image_ids_list, answers_list, questions_list = get_predictions(data_questions)
    accuracy, idx_wrong = evaluate(predictions_list, answers_list)
    print('Accuracy: ', accuracy)

    # save all predictions as json
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/predictions_testdev_balanced.json', 'w') as f:
        json.dump(predictions_list, f, indent=2)

    # save the wrong predictions as json
    with open('test_results/wrong_predictions_testdev_balanced.json', 'w') as f:
        wrong_predictions = [{'image_id': image_ids_list[i], 'question': questions_list[i], 'prediction': predictions_list[i]['prediction'], 'answer': answers_list[i]} for i in idx_wrong]
        json.dump(wrong_predictions, f, indent=2)