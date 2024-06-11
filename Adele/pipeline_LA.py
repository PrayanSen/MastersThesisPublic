import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import logging
import os
from typing import Dict
from transformers import (
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    AdapterType,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
    glue_tasks_num_labels,
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

    # Check for argument conflicts
    if (data_args.language_eval_data_file is None and la_args.eval_language_adapter) or (data_args.task_data_files is None and ta_args.eval_task_adapter) or (data_args.task_data_files is None and ft_args.eval_finetuning):
        raise ValueError("Cannot do evaluation without an evaluation data file.")
    if (la_args.train_language_adapter and data_args.language_train_data_file is None) or (ta_args.train_task_adapter and data_args.task_data_files is None) or (ft_args.do_finetuning and data_args.task_data_files is None):
        raise ValueError("Cannot do training without an training data file.")
    if (la_args.train_language_adapter and la_args.load_language_adapter is not None) or (ta_args.train_task_adapter and ta_args.load_task_adapter is not None):
        raise ValueError("Adapter must be trained OR loaded. Both is not possible.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    print()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(la_args)

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


    print("\n\n\n-----------------------------------------------\n---------------Language Adapter----------------\n-----------------------------------------------")

    # If a new language adapter should be created
    if la_args.load_language_adapter is None:  

        # Create language adapter
        print("\n\nCreate language adapter and freeze other layers for adapter training...")
        if la_args.train_language_adapter:
            base_model = getattr(model, model.base_model_prefix, model)
            # only add adapter to model if there is no adapter with same name
            if la_args.language_adapter not in base_model.config.adapters.adapter_list(AdapterType.text_lang):
                base_model.set_adapter_config(AdapterType.text_lang, la_args.language_adapter_config)
                base_model.add_adapter(la_args.language_adapter, AdapterType.text_lang)
            # enable adapter training and freeze other model weights
            base_model.train_adapter([la_args.language_adapter])
        print("...done")

        # Get datasets for language adapter training
        print("\n\nGet train dataset for language adapter...")
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.language_train_data_file, block_size=data_args.block_size) if la_args.train_language_adapter else None
        print("...done\nGet eval dataset for language adapter...")
        eval_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.language_eval_data_file, block_size=data_args.block_size) if la_args.eval_language_adapter else None
        print("...done")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

        # Update training_args with la_args to create Trainer
        training_args.do_train = la_args.train_language_adapter
        training_args.do_eval = la_args.eval_language_adapter
        training_args.per_device_train_batch_size = la_args.language_adapter_batch_size
        training_args.per_device_eval_batch_size = la_args.language_adapter_batch_size
        training_args.learning_rate = la_args.language_adapter_learning_rate
        training_args.num_train_epochs = la_args.language_adapter_epochs
        # Initialize Trainer
        print("\n\nTrain language adapter...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
            adapter_names=[[la_args.language_adapter]],
        )
        # train adapter 
        if la_args.train_language_adapter:
            trainer.train()
            print("...done")

            # if 2-sided-separated CDA: do second language adapter training
            if la_args.la_cda_type == "cda_sep": 

                print("\n\n\n---------------------------------------------------------------\n---------------Second Language Adapter Training----------------\n---------------------------------------------------------------")

                model.train_adapter([la_args.language_adapter])
                model.set_active_adapters(la_args.language_adapter)

                # Get datasets for second language adapter training
                print("\n\nGet train dataset for second language adapter training...")
                train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.language_train_data_file_cda_sep, block_size=data_args.block_size)
                print("...done")

                # Initialize Trainer
                print("\n\nStart second language adapter training...")
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    prediction_loss_only=True,
                    adapter_names=[[la_args.language_adapter]],
                )
                # train adapter 
                trainer.train()
                print("...done")

            # save adapter to output folder and to adapters folder
            model.save_adapter("{0}/language_adapter/".format(training_args.output_dir), la_args.language_adapter)
            if not os.path.exists("./language_adapters/{0}".format(la_args.language_adapter)):
                os.makedirs("./language_adapters/{0}".format(la_args.language_adapter))
            model.save_adapter("./language_adapters/{0}".format(la_args.language_adapter), la_args.language_adapter)


        # Evaluation
        if training_args.do_eval:
            print("\n\nDo language adapter evaluation...")
            logger.info("*** Evaluate ***")
            # Evaluate adapter and save the loss to output file
            eval_output = trainer.evaluate()
            result = {"loss": eval_output["eval_loss"]}
            output_eval_file = os.path.join("{0}/evaluations/".format(training_args.output_dir), "evaluation_results_language_adapter.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        print("...done")

    else:
        print("\n\nLanguage adapter training is skipped because already trained language adapter exists and will be loaded in next step.\n\n")

        # Load language adapter
        print("\n\nLoad language adapter...")
        model.load_adapter("./language_adapters/{0}".format(la_args.language_adapter))
        model.set_active_adapters(la_args.language_adapter)
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
