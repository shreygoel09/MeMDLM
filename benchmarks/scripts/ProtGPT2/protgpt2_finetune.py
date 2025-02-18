import pandas as pd
import os
import sys
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM


# Format sequence inputs based on ProtGPT fine-tuning requirements
def modify_sequences(sequence):
  modified_sequence = sequence.upper()
  modified_sequence = '\n'.join([modified_sequence[i:i+60] for i in range(0, len(modified_sequence), 60)])

  fasta = "<|endoftext|>"
  modified_sequence = fasta + "\n" + modified_sequence

  return modified_sequence

# Function to save sequences to txt files
def to_txt_file(df, filename):
  with open(filename, 'w') as f:
    for sequence in df['Sequence']:
      f.write(sequence + '\n')


# Modify the sequences
path = "/workspace/sg666/MeMDLM/MeMDLM"

train = pd.read_csv(path + "/data/membrane/train.csv")
test = pd.read_csv(path + "/data/membrane/test.csv")

#train = pd.concat([train, val])

train['Sequence'] = train['Sequence'].apply(modify_sequences)
test['Sequence'] = test['Sequence'].apply(modify_sequences)

print(train)
print(test)


# Save the modified sequences as txt files
train_data_path = path + '/benchmarks/results/de_novo/protgpt/train.txt'
test_data_path = path + '/benchmarks/results/de_novo/protgpt/test.txt'
to_txt_file(train, train_data_path)
to_txt_file(test, test_data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2").to(device)
print(model)

sys.stdout.flush()

finetune_protgpt2_command = [
    "python", "run_clm.py",
    "--model_name_or_path", "nferruz/ProtGPT2",
    "--train_file", "/workspace/sg666/MeMDLM/MeMDLM/benchmarks/results/de_novo/protgpt/train.txt",
    "--validation_file", "/workspace/sg666/MeMDLM/MeMDLM/benchmarks/results/de_novo/protgpt/test.txt",
    "--tokenizer_name", "nferruz/ProtGPT2",
    "--num_train_epochs", "5",
    "--logging_steps", "10",
    "--logging_dir", "test",
    "--do_train",
    "--do_eval",
    "--output_dir", "/workspace/sg666/MeMDLM/MeMDLM/benchmarks/scripts/ProtGPT2/finetuned_models/",
    "--overwrite_output_dir",
    "--learning_rate", "5e-4",
    "--per_device_train_batch_size", "2",
    "--evaluation_strategy", "epoch"
]

try:
    result = subprocess.run(finetune_protgpt2_command, check=True, text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print("Command failed with the following error:")
    print(e.stderr)  # Print standard error output
    print("Command output:")
    print(e.stdout)  # Print standard output if needed

