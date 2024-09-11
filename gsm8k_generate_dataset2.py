import openai
import numpy as np 
import json
from datasets import load_dataset
import tqdm


# dataset = load_dataset("gsm8k", "main")
# train_questions = np.array(dataset["train"]["question"])
# train_answers = np.array(dataset["train"]['answer'])

# test_questions = np.array(dataset["test"]["question"])
# test_answers = np.array(dataset["test"]['answer'])


with open("/data/katie_kang/openai_key_file_rail.txt", "r") as f:
    openai.api_key = f.read().strip()

def query_gpt4(prompt, model="gpt-4o", temp=0):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                # {"role": "system", "content": "Answer Question 3 using the same format as Answer 1 and 2."},
                
                
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,  # You can adjust the max tokens as needed
            temperature=temp,  # Adjust the creativity level
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


for num_iterations in range(5):

    responses = []

    for train_question_idx in tqdm.tqdm(range(len(train_questions))):

#         # prompt = f'''
#         # Question1: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
#         # Answer1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18

#         # Question2: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
#         # Answer2: It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3

#         # Question3: {train_questions[train_question_idx]}'''
        
#         # prompt = f'''
#         # Question1: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
#         # Answer1: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18

#         # Question2: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
#         # Answer2: It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3

#         # Question3: {train_questions[train_question_idx]}'''
        
        
            prompt = f'''Please act as a professional math teacher.
        Your goal is to create high quality math word problems to help students learn math.
        You will be given a math question. Please create a new question based on the Given Question and following
        instructions.
        To achieve the goal, you have one job.
        # Please generate a similar but new question according to the Given Question.
        You have four principles to do this.
        # Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and can be
        answered with only a number(float or integer). For example, DO NOT ask, ‘what is the amount of A, B and
        C?’.
        # Ensure the new question is in line with common sense of life. For example, the amount someone has or pays
        must be a positive number, and the number of people must be an integer.
        # Ensure your student can answer the new question without the given question. If you want to use some
        numbers, conditions or background in the given question, please restate them to ensure no information is
        omitted in your new question.
        # You only need to create the new question. Please DO NOT solve it.
        Given Question: {train_questions[train_question_idx]}
        Your output should be in the following format:
        CREATED QUESTION: <your created question>'''

#         prompt = f'''You will be given a math question and its solution. Please rephrase the solution in a different way.
#     You have two principles to do this.
#     # Ensure the rephrased solution is accurate and clear.
#     # Ensure the rephrased solution is in a different form than the original solution.
#     The format of your answer should follow these guidelines:
#     # Each step of your solution is separated by a new line.
#     # Each arithmetic operation is is highlighted by the symbol <<>>, e.g. 60/100 * 5 = <<60/100*5=3>>3
#     # The output ends with the final answer, e.g. #### 64
#     Below is an example of a question, its solution, and a rephrased solution which you can use as a reference.
#     Example question: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
#     Example solution: He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\n#### 624
#     Example rephrase of solution: James writes 3 pages per letter and writes to 2 friends, so he writes 3 * 2 = <<3*2=6>>6 pages per session.\nHe writes twice a week, so he writes 6 * 2 = <<6*2=12>>12 pages per week.\nThere are 52 weeks in a year, so he writes 12 * 52 = <<12*52=624>>624 pages a year.\n#### 624
    
#     Given question: {train_questions[train_question_idx]}
#     Given answer: {train_answers[train_question_idx]}
#     Your rephrase of of solution: '''

#         response = query_gpt4(prompt, temp=1)

#         print(train_questions[train_question_idx])
#         print("\nGPT4")
#         print(response.replace("\n\n", "\n"))


#         print("\nGround Truth")
#         print(train_answers[train_question_idx])
        
#         # responses.append(response.replace("CREATED QUESTION: ", ""))

#         responses.append(response.replace("\n\n", "\n"))
        
#     np.save(f"gsm8k_train_answers_gpt4o{num_iterations}.npy", responses)


# np.save("gsm8k_train_answers_gpt4o.npy", responses)

train_questions = np.load("gsm8k_train_questions_new_gpt4o.npy")

responses=[]
for question_idx in tqdm.tqdm(range(len(train_questions))):

    prompt = f'''Please act as a professional math teacher.
Your goal is to accurately solve a math word problem.
To achieve the goal, you have two jobs.
# Write detailed solution to a Given Question.
# Write the final answer to this question.
You have two principles to do this.
# Ensure the solution is step-by-step.
# Ensure the final answer is just a number (integer).
The format of your answer should follow these guidelines:
# Each step of your solution is separated by a new line.
# Each arithmetic operation is is highlighted by the symbol <<>>, e.g. 60/100 * 5 = <<60/100*5=3>>3
# The output ends with the final answer, e.g. #### 64
Below is an example of a question and its solution which you can use as a reference.
Example question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Example answer: He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 meters\n#### 540

Your given question: {train_questions[question_idx]}
Your answer: '''

    response = query_gpt4(prompt)

    print(question_idx)
    print(train_questions[question_idx])
    print("\nGPT4")
    print(response.replace("\n\n", "\n"))


    # print("\nGround Truth")
    # print(train_answers[question_idx])
    
    responses.append(response.replace("\n\n", "\n"))

import IPython; IPython.embed()