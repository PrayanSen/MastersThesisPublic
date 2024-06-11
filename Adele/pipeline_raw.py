import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import logging
import os
from typing import Dict
from transformers import (
    AdapterType,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
    TrainingArguments,
)

from arguments import ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments
from dataset_processing import TextDataset
from bias_becpro import evaluate_becpro
from bias_disco import evaluate_adjusted_disco


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in arguments.py and src/transformers/training_args.py
    # Parse the arguments and create data classes
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments, TrainingArguments))
    model_args, data_args, la_args, ta_args, ft_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    print()
    logger.info(model_args)
    logger.info(data_args)

    # Create result folders
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        os.makedirs("{0}/evaluations".format(training_args.output_dir))
        if la_args.train_language_adapter:
            os.makedirs("{0}/language_adapter".format(training_args.output_dir))
    # Set seed
    set_seed(training_args.seed)


    # Load tokenizer
    print("\n\nLoad tokenizer...")
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("You need to specify which tokenizer to be loaded. Give a valid tokenizer_name or model_name_or_path.")
    print("...done")

    # Adjust block_size if too low or too high
    if la_args.train_language_adapter:
        if data_args.block_size <= 0:
            data_args.block_size = 128
        else:
            data_args.block_size = min(data_args.block_size, tokenizer.max_len)


    # Create model config
    print("\n\nCreate model config...")
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("No valid model_name_or_path.")
    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm flag.")
    print("...done")

    # Create model with language model head and specified model config
    print("\n\nCreate model with language model head...")
    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        raise ValueError("No valid model_name_or_path.")
    model.resize_token_embeddings(len(tokenizer))
    print("...done")


    print("\n\n\n-----------------------------------------------\n----------------Bias Evaluation----------------\n-----------------------------------------------\n\n")

    # Bias evaluation if desired
    if data_args.bias_eval_task == "disco":
        # Evaluate on DisCo
        print("Evaluate on DisCo ...")
        evaluate_adjusted_disco(model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    elif data_args.bias_eval_task == "bec-pro_english":
        # Evaluate on BEC-Pro
        print("Evaluate on English BEC-Pro ...")
        evaluate_becpro("english", model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    elif data_args.bias_eval_task == "bec-pro_german":
        # Evaluate on BEC-Pro
        print("Evaluate on English BEC-Pro ...")
        evaluate_becpro("german", model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    else:
        print("\n\n\nNo bias evaluation wished...")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
