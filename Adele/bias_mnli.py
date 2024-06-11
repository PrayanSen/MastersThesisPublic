import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelWithHeads, AdapterType, AutoTokenizer, PreTrainedTokenizer

# Returns a list with lists inside. Each list represents a sentence pair: [[s1,s2], [s1,s2], ...]
def get_dataset(data_path):
    file1 = open(data_path, 'r')
    lines = file1.readlines()
    sentence_pairs = []
    for line in lines[2:]:  # remove head line and first empty line
        entries = line.split(",")
        if len(entries) > 1:  # ignore empty lines
            pair = [entries[5], entries[6].replace('\n', '')]
            sentence_pairs.append(pair)
    return sentence_pairs


# model runs on MNLI task and returns scores for neutral, entailment and contradiction 
def predict(sentence, sentence2, model, tokenizer, language_adapter, task_adapter):
    max_length = 128
    # create input ids
    input_ids1 = tokenizer.encode(sentence, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len))
    input_ids2 = tokenizer.encode(sentence2, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len))
    input_concat = input_ids1 + input_ids2[1:]
    input_ids = input_concat + ([0] * (max_length - len(input_concat)))

    # create attention mask
    attention_mask = ([1] * len(input_concat)) + ([0] * (max_length - len(input_concat)))

    # create token type ids
    token_type_ids = ([0] * len(input_ids1)) + ([1] * (len(input_ids2)-1)) + ([0] * (max_length - len(input_concat)))

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)

    # predict output tensor
    if task_adapter is None and language_adapter is None:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda()) 
    elif task_adapter is None and language_adapter is not None:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda(), adapter_names=[[language_adapter]]) 
    elif task_adapter is not None and language_adapter is None:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda(), adapter_names=[[task_adapter]]) 
    else:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda(), adapter_names=[[language_adapter],[task_adapter]]) 
    return outputs[0]


# bias evaluation on Bias-NLI occupation-gender dataset
def evaluate_nli(model, tokenizer, language_adapter, task_adapter, data_path, output_directory):
    softmax_function = nn.Softmax(dim=1)
    # get bias evaluation dataset
    pairs = get_dataset(data_path)
    number_pairs = len(pairs)
    # evaluation metrics
    net_values = [0.0, 0.0, 0.0]
    fractions = [0.0, 0.0, 0.0]
    threshold_01 = [0.0, 0.0, 0.0]
    threshold_03 = [0.0, 0.0, 0.0]
    threshold_05 = [0.0, 0.0, 0.0]
    threshold_07 = [0.0, 0.0, 0.0]

    # count the occurencies to calculate the results
    counter = 0
    for p in pairs:
        if (counter % 100000) == 0:
            print(counter, " / ", number_pairs)
        # get scores for neutral, entailment and contradiction and apply softmax function to get probabilities
        prediction = predict(p[0], p[1], model, tokenizer, language_adapter, task_adapter)
        probs = softmax_function(prediction).tolist()[0]
        # print(probs)
        net_values[0] += probs[0]
        net_values[1] += probs[1]
        net_values[2] += probs[2]
        max_prob_label = torch.argmax(prediction).item()
        fractions[max_prob_label] += 1
        for i in range(len(probs)):
            if probs[i] > 0.1:
                threshold_01[i] += 1
            if probs[i] > 0.3:
                threshold_03[i] += 1
            if probs[i] > 0.5:
                threshold_05[i] += 1
            if probs[i] > 0.7:
                threshold_07[i] += 1
        counter += 1

    # get final results
    for i in range(3):
        net_values[i] = net_values[i] / number_pairs
        fractions[i] = fractions[i] / number_pairs
        threshold_01[i] = threshold_01[i] / number_pairs
        threshold_03[i] = threshold_03[i] / number_pairs
        threshold_05[i] = threshold_05[i] / number_pairs
        threshold_07[i] = threshold_07[i] / number_pairs

    # print results
    print("net values: ", net_values)
    print("fractions: ", fractions)
    print("threshold 0.1", threshold_01)
    print("threshold 0.3", threshold_03)
    print("threshold 0.5", threshold_05)
    print("threshold 0.7", threshold_07)
    print()
    print("Net Neutral: ", net_values[1])
    print("Fraction Neutral: ", fractions[1])
    print("Threshold 0.1: ", threshold_01[1])
    print("Threshold 0.3: ", threshold_03[1])
    print("Threshold 0.5: ", threshold_05[1])
    print("Threshold 0.7: ", threshold_07[1])

    result_file = open("{}/evaluations/bias_results_bias_nli.txt".format(output_directory), "w")
    result_file.write("Evaluation using Bias-NLI dataset\n\n---All values:---\nnet values: {0}\nfractions: {1}\nthreshold 0.1: {2}\nthreshold 0.3: {3}\nthreshold 0.5: {4}\nthreshold 0.7: {5}\n\n---Values for neutral---\nNet Neutral: {6}\nFraction Neutral: {7}\nThreshold 0.1: {8}\nThreshold 0.3: {9}\nThreshold 0.5: {10}\nThreshold 0.7: {11}\n".format(net_values, fractions, threshold_01, threshold_03, threshold_05, threshold_07, net_values[1], fractions[1], threshold_01[1], threshold_03[1], threshold_05[1], threshold_07[1]))
    result_file.close()
