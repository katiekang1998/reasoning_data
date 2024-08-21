from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_name = "ckpts/gsm8k_orig_12epochs_full_lr2e-06_bs128/checkpoint-464"  # Replace with the specific LLaMA model you want
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)

model.to("cuda")

model.eval()

def get_output(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda").input_ids
    output = model(input_ids)
    
    # attention_scores = outputs.attentions 
    return output


def get_sample(input_text):
    input_text = input_text+ "\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda").input_ids
    # outputs = model(input_ids)
    output = model.generate(input_ids, max_length=500, num_return_sequences=5, do_sample=True, temperature=0.8)
    print(output)
    print(input_ids)
    
    # attention_scores = outputs.attentions 
    for output_str in tokenizer.batch_decode(output):
        print(output_str)
        print("")


import IPython; IPython.embed(); exit(1)



input_text = 'Jordan has a new hit song on Spotify. 3 months are left in the year, and she currently has 60,000 listens. If the number of listens per month doubles, how many listens will the song have by the end of the year?'

input_text = 'Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?'


question = 'Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?'
question+= "\nAnswer:"
input_text =question +' He saved up $110 total because 95 + 15 = <<95+15=110>>110\nHe saved $15 from his allowance because 3 x 5 = <<3*5=15>>15\nHe earned $60 mowing lawns because 4 x 15 = <<4*15=60>>60\nHe earned $35 shoveling driveways because 110 - 60 - 15 = <<110-60-15=35>>35\nHe shoveled 5 driveways because 35 / 7 = <<35/7=5>>5\n#### 5'



question = 'To make pizza, together with other ingredients, Kimber needs 10 cups of water, 16 cups of flour, and 1/2 times as many teaspoons of salt as the number of cups of flour. Calculate the combined total number of cups of water, flour, and teaspoons of salt that she needs to make the pizza.'
question+= "\nAnswer:"
input_text =question +' To make the pizza, Kimber half as many teaspoons of salt as the number of cups of flour, meaning she needs 1/2*16 = <<16*1/2=8>>8 teaspoons of salt.\nThe total number of cups of flour and teaspoons of salt she needs is 8+16 = <<8+16=24>>24\nShe also needs 10 cups of water, which means the total number of cups of water and flour and teaspoons of salt she needs is 24+10 = <<24+10=34>>34\n#### 34'

output = get_output(input_text)


len(tokenizer(question).input_ids)


for i in range(len(output.attentions)):
    # argmax_idxs = (output.attentions[i][0][:, -1].max(axis=-1).indices)
    # print(((argmax_idxs > 0)*(argmax_idxs < output.attentions[i].shape[-1])).sum())
    
    print(output.attentions[i][0][:, 100 , 1:].max(axis=-1).indices)