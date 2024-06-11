import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import logging
import os
from typing import Dict
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr 
from transformers import (
    EvalPrediction,
    GlueDataset,
    AutoModelWithHeads,
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
from bias_mnli import evaluate_nli
from bias_stsb import evaluate_bias_sts


logger = logging.getLogger(__name__)


# evaluates on MNLI dev dataset and returns accuracy
def compute_metrics_mnli(pred: EvalPrediction) -> Dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# evaluates on STS-B dev dataset and returns correlation
def compute_metrics_stsb(pred: EvalPrediction) -> Dict:
    labels = pred.label_ids
    preds = []
    for p in pred.predictions:
        preds.append(p[0])
    corr, p = pearsonr(preds,labels)
    return {'accuracy': corr}


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
    if (ta_args.train_task_adapter and ta_args.task_adapter_task is None) or (ft_args.do_finetuning and ft_args.finetuning_task is None):
        raise ValueError("Cannot do training without a task.")
    if (la_args.train_language_adapter and la_args.load_language_adapter is not None) or (ta_args.train_task_adapter and ta_args.load_task_adapter is not None):
        raise ValueError("Adapter must be trained OR loaded. Both is not possible.")
    if (ta_args.train_task_adapter and ft_args.do_finetuning):
        raise ValueError("Choose train_task_adapter OR do_finetuning, not both.")

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
    logger.info(ft_args)

    # Create result folders
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        os.makedirs("{0}/evaluations".format(training_args.output_dir))
        if ft_args.do_finetuning:
            os.makedirs("{0}/finetuned_model".format(training_args.output_dir))
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

    print("\n\n\n----------------------------------------------\n------------------Finetuning------------------\n----------------------------------------------\n\n") 

    # create data training arguments for glue task finetuning_task and update training_args
    glue_data_args = GlueDataTrainingArguments(task_name=ft_args.finetuning_task, data_dir=data_args.task_data_files)
    training_args.do_train = ft_args.do_finetuning
    training_args.do_eval = ft_args.eval_finetuning
    training_args.per_device_train_batch_size = ft_args.finetuning_batch_size
    training_args.per_device_eval_batch_size = ft_args.finetuning_batch_size
    training_args.learning_rate = ft_args.finetuning_learning_rate
    training_args.num_train_epochs = ft_args.finetuning_epochs
        
    # Create new model config
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("No valid model_name_or_path.")
    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm flag.")

    # Create model with head (this time no language model head) and specified model config
    print("\n\nLoad model with head...")
    if model_args.model_name_or_path:
        model = AutoModelWithHeads.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        raise ValueError("No valid model_name_or_path.")
    model.resize_token_embeddings(len(tokenizer))
    print("...done")

    # Add classification head
    print("\n\nAdd classification head...")
    model.add_classification_head(ft_args.finetuning_task, num_labels=glue_tasks_num_labels[glue_data_args.task_name])
    model.set_active_adapters(ft_args.finetuning_task)
    print("...done")

    # Get datasets
    print("\n\nGet task datasets...")
    train_dataset = GlueDataset(glue_data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(glue_data_args, tokenizer=tokenizer, mode='dev')
    print("...done")

    # Initialize Trainer for finetuning
    print("\n\nDo finetuning...")
    if glue_data_args.task_name == "mnli":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_mnli,
        )
    elif glue_data_args.task_name == "sts-b":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_stsb,
        )
    else:
        raise ValueError("Task name must be mnli or sts-b")

    # train model and save it to output folder and to model folder 
    trainer.train()
    model.save_pretrained('{0}/finetuned_model/'.format(training_args.output_dir))
    if not os.path.exists("./finetuned_models/{0}".format(training_args.output_dir[10:])):
        os.makedirs("./finetuned_models/{0}".format(training_args.output_dir[10:]))
    model.save_pretrained('./finetuned_models/{0}'.format(training_args.output_dir[10:]))
    print("...done")


    # Evaluation
    if ft_args.eval_finetuning:
        print("\n\nDo evaluation...")
        logger.info("*** Evaluate ***")
        # Evaluate model and save the loss and accuracy/correlation to output file
        eval_output = trainer.evaluate()
        loss = eval_output["eval_loss"]
        accuracy = eval_output["eval_accuracy"]
        result = {"loss": loss, "accuracy": accuracy}
        output_eval_file = os.path.join("{0}/evaluations/".format(training_args.output_dir), "evaluation_results_finetuning.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        print("...done")


    print("\n\n\n-----------------------------------------------\n----------------Bias Evaluation----------------\n-----------------------------------------------")
 
    # Bias evaluation if desired
    if data_args.bias_eval_data_file is not None: 

        print("\n\nDo bias evaluation...")
        if glue_data_args.task_name == "mnli":
            # Evaluate on Bias-NLI dataset
            print("Evaluate on Bias-NLI dataset...")
            evaluate_nli(model=model.cuda(), tokenizer=tokenizer, language_adapter=la_args.language_adapter, task_adapter=ta_args.task_adapter, data_path=data_args.bias_eval_data_file, output_directory=training_args.output_dir)
        elif glue_data_args.task_name == "sts-b":
            # Evaluate on CDA sts-b dataset
            print("Evaluate on CDA sts-b dataset ...")
            evaluate_bias_sts(model=model.cuda(), tokenizer=tokenizer, language_adapter=la_args.language_adapter, task_adapter=ta_args.task_adapter, data_path=data_args.bias_eval_data_file, output_directory=training_args.output_dir)
        else:
            raise ValueError("Task must be mnli or sts-b")

        print("...done")
    else:
        print("\n\nNo bias evaluation wished...")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
