sft_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request." \
    "\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"


all_prompt = {}

mindrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. Can you help me decide?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Here are the user's historical interactions: {inters}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Based on the items that the user has interacted with: {inters}, can you determine what item would be recommended to him next?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in order: {inters}. What else do you think the user need?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = "Here is the item interaction history of the user: {inters}, what to recommend to the user next?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = "Which item would the user be likely to interact with next after interacting with items {inters}?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = "After interacting with items {inters}, what is the next item that could be recommended for the user?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "Given the user's historical interactive items arranged in chronological order: {inters}, can you recommend a suitable item for the user?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "Considering the user has interacted with items {inters}. What is the next recommendation for the user?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "What is the top recommended item for the user who has previously interacted with items {inters} in order?"
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——12
prompt = {}
prompt["instruction"] = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user most desires based on the given interaction records."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. The historical interactions are provided as follows: {inters}."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = "You can access the user's historical item interaction records: {inters}. Now your task is to recommend the next potential item to him, considering his past interactions."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = "You have observed that the user has interacted with the following items: {inters}, please recommend a next item that you think would be suitable for the user."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. Using this history as a reference, please select the next item to recommend to the user."
prompt["response"] = "{c1}{c2}{c3}-{item}"
mindrec_prompt.append(prompt)

all_prompt["mindrec"] = mindrec_prompt
