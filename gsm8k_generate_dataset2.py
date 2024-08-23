import openai
import numpy as np 
import json
from datasets import load_dataset
import tqdm


dataset = load_dataset("gsm8k", "main")
train_questions = np.array(dataset["train"]["question"])
train_answers = np.array(dataset["train"]['answer'])

test_questions = np.array(dataset["test"]["question"])
test_answers = np.array(dataset["test"]['answer'])


with open("/data/katie_kang/openai_key_file_rail.txt", "r") as f:
    openai.api_key = f.read().strip()

def query_gpt4(prompt, model="gpt-4o"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": "Answer Question 3 using the same format as Answer 1 and 2."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,  # You can adjust the max tokens as needed
            temperature=0.,  # Adjust the creativity level
        )
        
        response_content = response['choices'][0]['message']['content']

        # Get usage data
        usage = response['usage']
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']
        total_tokens = usage['total_tokens']
        
        cost = usage['total_tokens'] *5 / 1000000
        
        # print(f"Prompt tokens: {prompt_tokens}")
        # print(f"Completion tokens: {completion_tokens}")
        # print(f"Total tokens: {total_tokens}")
        # print(f"Cost: ${cost}")
        
        
        
        
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Example usage


responses = []

for train_question_idx in tqdm.tqdm(range(1000)):

    # prompt = f'''
    # Question1: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    # Answer1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18

    # Question2: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
    # Answer2: It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3

    # Question3: {train_questions[train_question_idx]}'''
    
    prompt = f'''
    Question1: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    Answer1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18

    Question2: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
    Answer2: It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3

    Question3: {train_questions[train_question_idx]}'''
    response = query_gpt4(prompt)

    # print(train_questions[train_question_idx])
    # print("\nGPT4")
    # print(response)


    # print("\nGround Truth")
    # print(train_answers[train_question_idx])
    
    responses.append(response)



import IPython; IPython.embed(); exit(1)