from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# Load dataset
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)

# print(dataset)

# Load pretrained model
# Load the pre-trained [FLAN-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5) and its tokenizer directly from HuggingFace.
# Notice that you will be using the [small version](https://huggingface.co/google/flan-t5-base) of FLAN-T5.
# Setting `torch_dtype=torch.bfloat16` specifies the memory type to be used by this model.

model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# It is possible to pull out the number of model parameters and find out how many of them are trainable.
# The following function can be used to do that, at this stage, you do not need to go into details of it. 

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()

    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

# Test the model with the zero shot inferencing. You can see that the model struggles to summarize the dialogue compared to the baseline summary,
# but it does pull out some important information from the text which indicates the model can be fine-tuned to the task at hand.
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"], 
        max_new_tokens=200,
    )[0], 
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

# perform FULL FINE-TUNING

# Preprocess the Dialog-Summary Dataset
# You need to convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM
# Then preprocess the prompt-response dataset into tokens and pull out their `input_ids` (1 per token).

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])

# To save some time in the lab, you will subsample the dataset:
tokenized_datasets = tokenized_datasets.filter(lambda _example, index: index % 100 == 0, with_indices=True)

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

# Fine-Tune the Model with the Preprocessed Dataset
# Now utilize the built-in Hugging Face `Trainer` class (see the documentation [here](https://huggingface.co/docs/transformers/main_classes/trainer)).
# Pass the preprocessed dataset with reference to the original model.
# Other training parameters are found experimentally and there is no need to go into details about those at the moment.
perform_train_model = False
if perform_train_model:
    output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=1
    )

    trainer = Trainer(
        model=original_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )

    trainer.train()

# Download an instruct model instead
use_downloaded_instruct_model = True
if use_downloaded_instruct_model:
    instruct_model_name='truocpham/flan-dialogue-summary-checkpoint'

    instruct_model = AutoModelForSeq2SeqLM.from_pretrained(instruct_model_name, torch_dtype=torch.bfloat16)

use_local_instruct_model = False
if use_local_instruct_model:
    instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint-local", torch_dtype=torch.bfloat16)

# Evaluate the Model Qualitatively (Human Evaluation)
# As with many GenAI applications, a qualitative approach where you ask yourself the question "Is my model behaving the way it is supposed to?" is usually a good starting point.
# In the example below (the same one we started this notebook with), you can see how the fine-tuned model is able to create a reasonable summary
# of the dialogue compared to the original inability to understand what is being asked of the model.
run_evaluation = False
if run_evaluation:
    index = 200
    dialogue = dataset['test'][index]['dialogue']
    human_baseline_summary = dataset['test'][index]['summary']

    prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
    print(dash_line)
    print(f'ORIGINAL MODEL:\n{original_model_text_output}')
    print(dash_line)
    print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')

# Evaluate the Model Quantitatively (with ROUGE Metric)
# The [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)) helps quantify the validity of summarizations produced by models.
# It compares summarizations to a "baseline" summary which is usually created by a human.
# While not perfect, it does indicate the overall increase in summarization effectiveness that we have accomplished by fine-tuning.
run_rouge_metric = False
if run_rouge_metric:
    rouge = evaluate.load('rouge')

    # Generate the outputs for the sample of the test dataset (only 10 dialogues and summaries to save time), and save the results.
    dialogues = dataset['test'][0:10]['dialogue']
    human_baseline_summaries = dataset['test'][0:10]['summary']

    original_model_summaries = []
    instruct_model_summaries = []

    print('type of dialogues:\t', type(dialogues))

    for idx, dialogue in enumerate(dialogues):
        print('...working on ', idx)
        prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
        original_model_summaries.append(original_model_text_output)

        instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
        instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
        instruct_model_summaries.append(instruct_model_text_output)
        
    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))
    
    df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries'])
    print(df)

    # Evaluate the models computing ROUGE metrics. Notice the improvement in the results!
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_summaries,
        references=human_baseline_summaries[0:len(instruct_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL:')
    print(instruct_model_results)

# The file `data/dialogue-summary-training-results.csv` contains a pre-populated list of all model results
# which you can use to evaluate on a larger section of data. Let's do that for each of the models:
available_populated_list_evaluation = False
if available_populated_list_evaluation:
    rouge = evaluate.load('rouge')

    results = pd.read_csv("dialogue-summary-training-results.csv")

    human_baseline_summaries = results['human_baseline_summaries'].values
    original_model_summaries = results['original_model_summaries'].values
    instruct_model_summaries = results['instruct_model_summaries'].values

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_summaries,
        references=human_baseline_summaries[0:len(instruct_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL:')
    print(instruct_model_results)

    # The results show substantial improvement in all ROUGE metrics:
    print("Absolute percentage improvement of INSTRUCT MODEL over HUMAN BASELINE")

    improvement = (np.array(list(instruct_model_results.values())) - np.array(list(original_model_results.values())))
    for key, value in zip(instruct_model_results.keys(), improvement):
        print(f'{key}: {value*100:.2f}%')

# Perform Parameter Efficient Fine-Tuning (PEFT)
# Now, let's perform **Parameter Efficient Fine-Tuning (PEFT)** fine-tuning as opposed to "full fine-tuning" as you did above.
# PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning - with comparable evaluation results as you will see soon. 
# PEFT is a generic term that includes **Low-Rank Adaptation (LoRA)** and prompt tuning (which is NOT THE SAME as prompt engineering!).
# In most cases, when someone says PEFT, they typically mean LoRA. LoRA, at a very high level, allows the user to fine-tune their model using fewer compute resources (in some cases, a single GPU).
# After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges.
# This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).  
# That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request.
# The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.
        
# Setup the PEFT/LoRA model for Fine-Tuning

# You need to set up the PEFT/LoRA model for fine-tuning with a new layer/parameter adapter.
# Using PEFT/LoRA, you are freezing the underlying LLM and only training the adapter.
# Have a look at the LoRA configuration below. Note the rank (`r`) hyper-parameter, which defines the rank/dimension of the adapter to be trained.
using_perf = False
if using_perf:
    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )

    # Add LoRA adapter layers/parameters to the original LLM to be trained.
    peft_model = get_peft_model(original_model, 
                            lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    # Train PEFT Adapter
    # Define training arguments and create `Trainer` instance.
    output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning.
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1    
    )
        
    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Now everything is ready to train the PEFT adapter and save the model.
    peft_trainer.train()

    peft_model_path="./peft-dialogue-summary-checkpoint-local"

    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

# Use pretrained PEFT
use_peft_checkpoint = True
if use_peft_checkpoint:
    peft_dialogue_summary_checkpoint = 'intotheverse/peft-dialogue-summary-checkpoint'
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                        peft_dialogue_summary_checkpoint, 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    # The number of trainable parameters will be `0` due to `is_trainable=False` setting:
    print(print_number_of_trainable_model_parameters(peft_model))

use_peft_local_checkpoint = False
if use_peft_local_checkpoint:
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    peft_model_path="./peft_dialogue_summary_checkpoint_local/"
    peft_model = PeftModel.from_pretrained(peft_model_base, 
                                        peft_model_path, 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    print(print_number_of_trainable_model_parameters(peft_model))

# Evaluate the Model Qualitatively (Human Evaluation)
# Make inferences for the same example as in sections [1.3](#1.3) and [2.3](#2.3), with the original model, fully fine-tuned and PEFT model.
evaluate_peft_model = True
if evaluate_peft_model:
    human_baseline_summary = dataset['test'][index]['summary']
    index = 200
    dialogue = dataset['test'][index]['dialogue']
    baseline_human_summary = dataset['test'][index]['summary']

    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary: """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
    print(dash_line)
    print(f'ORIGINAL MODEL:\n{original_model_text_output}')
    print(dash_line)
    print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
    print(dash_line)
    print(f'PEFT MODEL: {peft_model_text_output}')
