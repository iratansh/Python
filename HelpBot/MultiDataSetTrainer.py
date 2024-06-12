"""
MultiDatasetTrainer class for training a model on multiple datasets.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

class MultiDatasetTrainer:
    """
    Initialize the MultiDatasetTrainer with a model.
    """
    def __init__(self, model_name='gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def load_dataset(self, dataset_name, split='train'):
        """
        Load a dataset from HuggingFace.
        Input: dataset_name (str), split (str)
        Output: dataset (DatasetDict)
        """
        try:
            dataset = load_dataset(dataset_name, split=split)
            logging.info(f"Loaded dataset {dataset_name} successfully.")
            return dataset
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def preprocess_dataset(self, dataset):
        """
        Preprocess the dataset by tokenizing it.
        Input: dataset (DatasetDict)
        Output: tokenized dataset (DatasetDict)
        """
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, datasets, output_dir='output', epochs=3):
        """
        Train the model on the combined datasets.
        Input: datasets (list of str), output_dir (str), epochs (int)
        Output: None
        """
        combined_dataset = None
        for dataset_name in datasets:
            dataset = self.load_dataset(dataset_name)
            if dataset:
                dataset = self.preprocess_dataset(dataset)
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    combined_dataset = combined_dataset.concatenate(dataset)

        if combined_dataset:
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                save_steps=10_000,
                save_total_limit=2,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=combined_dataset,
                data_collator=data_collator,
            )

            trainer.train()
        else:
            logging.error("No datasets were loaded successfully. Fix the issue.")
