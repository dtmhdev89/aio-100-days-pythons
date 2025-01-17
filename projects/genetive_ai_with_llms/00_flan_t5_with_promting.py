from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

load_dotenv()

# os.environ["HF_TOKEN"] = os.getenv('HF_TOKEN')
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

# for i, index in enumerate(example_indices):
#     print(dash_line)
#     print('Example ', i + 1)
#     print(dash_line)
#     print('INPUT DIALOGUE:')
#     print(dataset['test'][index]['dialogue'])
#     print(dash_line)
#     print('BASELINE HUMAN SUMMARY:')
#     print(dataset['test'][index]['summary'])
#     print(dash_line)
#     print()

# Load the FLAN-T5 model, creating an instance of the AutoModelForSeq2SeqLM class with the .from_pretrained() method.
model_name='google/flan-t5-base'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# To perform encoding and decoding, you need to work with text in a tokenized form. Tokenization is the process of splitting texts into smaller units that can be processed by the LLM models.
# Download the tokenizer for the FLAN-T5 model using AutoTokenizer.from_pretrained() method. Parameter use_fast switches on fast tokenizer. At this stage, there is no need to go into the details of that, but you can find the tokenizer parameters in the documentation.
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Test the tokenizer encoding and decoding a simple sentence:
sentence = "What time is it, Tom?"

sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0],
        skip_special_tokens=True
    )

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)

# Now it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering. Prompt engineering is an act of a human changing the prompt (input) to improve the response for a given task.
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')

# Summarize Dialogue with an Instruction Prompt

## Zero shot inference with instruct prompt

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
    Summarize down the following conversation.

    {dialogue}

    """

    # Input constructed prompt instead of the dialogue.
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

## Zero shot inference with prompt template from FLAN-T5

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
    Dialogue:

    {dialogue}

    What was going on?
    """

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

## One shot inference
    
def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
        Dialogue:

        {dialogue}

        What was going on?
        {summary}

        """

    dialogue = dataset['test'][example_index_to_summarize]['dialogue']

    prompt += f"""
    Dialogue:

    {dialogue}

    What was going on?
    """

    return prompt

example_indices_full = [40]
example_index_to_summarize = 200

one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(one_shot_prompt)

summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')

## Few shot inference
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200

few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(few_shot_prompt)

summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')

# generation_config = GenerationConfig(max_new_tokens=50)
# generation_config = GenerationConfig(max_new_tokens=10)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)

# inputs = tokenizer(few_shot_prompt, return_tensors='pt')
# output = tokenizer.decode(
#     model.generate(
#         inputs["input_ids"],
#         generation_config=generation_config,
#     )[0],
#     skip_special_tokens=True
# )

# print(dash_line)
# print(f'MODEL GENERATION - FEW SHOT:\n{output}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
