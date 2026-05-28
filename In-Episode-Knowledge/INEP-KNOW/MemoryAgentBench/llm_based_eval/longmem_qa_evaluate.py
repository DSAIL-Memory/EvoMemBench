import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np
from datasets import load_dataset, load_from_disk

import dotenv
dotenv.load_dotenv()


#@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt


def load_references_from_huggingface(huggingface_dataset_name, source_dataset_name):
    """
    Load and process reference data from a Hugging Face dataset.

    Args:
        huggingface_dataset_name (str): The name of the Hugging Face dataset.
        source_dataset_name (str): The source name to filter by in the dataset's metadata.

    Returns:
        list: A list of dictionaries, where each dictionary represents a question-answer pair.
    """
    print(f"Loading data from Hugging Face dataset: {huggingface_dataset_name}, split: Accurate_Retrieval")
    # Load the full dataset for the 'Accurate_Retrieval' split
    full_dataset = load_dataset(huggingface_dataset_name, split='Accurate_Retrieval', revision="main")

    # Filter the dataset to get entries matching the specified dataset source
    print(f"Filtering for source: {source_dataset_name}")
    filtered_dataset = full_dataset.filter(lambda example: example['metadata']['source'] == source_dataset_name)

    # Process the filtered data to create a flat list of question-answer pairs
    # This is necessary because each entry in the Hugging Face dataset can contain multiple QA pairs.
    references = []
    for entry in filtered_dataset:
        # Ensure all lists within an entry have the same number of items
        num_questions = len(entry['questions'])
        if not (num_questions == len(entry['answers']) and \
                num_questions == len(entry['metadata']['question_ids']) and \
                num_questions == len(entry['metadata']['question_types'])):
            print(f"Warning: Skipping entry due to mismatched lengths in QA data. Question IDs: {entry['metadata']['question_ids']}")
            continue

        # Unpack each QA pair into a separate dictionary
        for i in range(num_questions):
            references.append({
                'question': entry['questions'][i],
                'answer': entry['answers'][i],
                'question_id': entry['metadata']['question_ids'][i],
                'question_type': entry['metadata']['question_types'][i],
                'context': entry['context']  # Preserve context
            })
    print(f"Loaded and processed {len(references)} references from source '{source_dataset_name}'.")
    return references


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--evaluated_method', type=str, default='gpt-4o-mini')
    parser.add_argument('--huggingface_dataset_name', type=str, default="ai-hyz/MemoryAgentBench")
    parser.add_argument('--dataset', type=str, default='longmemeval_s*')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    args = parser.parse_args()

    verbose = True
    metric_model="gpt-4o"
    _env_vals = dotenv.dotenv_values()
    metric_client = OpenAI(
        api_key=_env_vals.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=_env_vals.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    hyp_folder = f'./outputs/{args.evaluated_method}/Accurate_Retrieval'
    
    # make the output dir
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    
    ## find the json file in the folder
    print('Evaluating method:', args.evaluated_method)
    if args.dataset == 'longmemeval_s':
        for root, _, files in os.walk(hyp_folder):
            for file in files:
                if file.endswith('.json') and 'longmemeval_s_' in file and "*" not in file:
                    hyp_file = os.path.join(root, file)
        with open(hyp_file, 'r', encoding='utf-8') as f:
            hypotheses = (json.load(f))["data"]
        
        hyper_file_tag = hyp_file.split('/')[-1].split('.')[0]
        result_file = os.path.join(args.output_dir, '.eval-results-{}-{}'.format(args.evaluated_method, hyper_file_tag))
        references = load_references_from_huggingface(args.huggingface_dataset_name, args.dataset)
    elif args.dataset in ('longmemeval_s*', 'longmemeval_star'):          
        for root, _, files in os.walk(hyp_folder):
            for file in files:
                if 'longmemeval_s*_' in file:
                    hyp_file = os.path.join(root, file)
        
        print('Hypothesis file:', hyp_file)
        with open(hyp_file, 'r', encoding='utf-8') as f:
            hypotheses = (json.load(f))["data"]
        
        hyper_file_tag = hyp_file.split('/')[-1].split('.')[0]
        result_file = os.path.join(args.output_dir, '.eval-results-{}-{}'.format(args.evaluated_method, hyper_file_tag))
        references = load_references_from_huggingface(args.huggingface_dataset_name, 'longmemeval_s*')

    ### make sure every question from references and hypotheses are the same   
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}

    if not os.path.exists(result_file):
        with open(result_file, 'w') as out_f:
            logs = []
            total_eval_input_tokens = 0
            total_eval_output_tokens = 0
            for idx, entry in tqdm(enumerate(references), total=len(references)):
                if entry['question_id'] not in qid2qtype:
                    print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
                    continue

                qtype = qid2qtype[entry['question_id']]
                q = qid2qdata[entry['question_id']]['question']
                ans = qid2qdata[entry['question_id']]['answer']
                hyp = hypotheses[idx]['output']
                ans2= hypotheses[idx]['answer']
                if ans2 != ans:
                    print("ans2 != ans, please check the data.")
                    print('Reference answer:', ans)
                    print('Hypothesis answer:', ans2)
                    raise ValueError('Answer in the hypothesis does not match the reference answer. Please check the data.')

                prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
                kwargs = {
                    'model': metric_model,
                    'messages':[
                        {"role": "user", "content": prompt}
                    ],
                    'n': 1,
                    'temperature': 0,
                    'max_tokens': 10
                }
                completion = chat_completions_with_backoff(metric_client, **kwargs)
                total_eval_input_tokens += completion.usage.prompt_tokens
                total_eval_output_tokens += completion.usage.completion_tokens
                eval_response = completion.choices[0].message.content.strip()
                label = 'yes' in eval_response.lower()
                entry['autoeval_label'] = {
                    'model': metric_model,
                    'label': label
                }
                ## entry without context
                entry['context'] = None
                logs.append(entry)
                if verbose:
                    print(json.dumps({
                        'question': q,
                        'answer': ans,
                        'hypothesis': hyp,
                        'autoeval_label': label
                    }, indent=4), flush=True)
                print(json.dumps(entry), file=out_f)
                qtype2acc[qid2qtype[entry['question_id']]].append(1 if label else 0)

        print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
        for k,v in qtype2acc.items():
            print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))
        print(f"Eval input tokens: {total_eval_input_tokens}, Eval output tokens: {total_eval_output_tokens}")

        print('Saved to', result_file)
    else:
        print('Result file already exists. Skipping evaluation.')
        with open(result_file, 'r') as out_f:
            logs = [json.loads(line) for line in out_f.readlines()]
        print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
        for log in logs:
            qtype2acc[qid2qtype[log['question_id']]].append(1 if log['autoeval_label']['label'] else 0)
            
        for k,v in qtype2acc.items():
            print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))
