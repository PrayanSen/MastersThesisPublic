import torch
from transformers import AutoModelForSequenceClassification, AutoModelWithHeads, AdapterType, AutoTokenizer, PreTrainedTokenizer

# Returns a list with lists inside. Each list represents a sentence pair: [[s1,s2], [s1,s2], ...]
def get_dataset_bias_sts(data_path):
    file1 = open(data_path, 'r', encoding="utf-8")
    lines = file1.readlines()
    sentence_pairs = []
    for line in lines: 
        entries = line.split("\t")
        if len(entries) > 1:  # ignore empty lines
            pair = [entries[0].replace('\n', ''), entries[1].replace('\n', ''), entries[2].replace('\n', ''), entries[3].replace('\n', '')]
            sentence_pairs.append(pair)
    return sentence_pairs


# model runs on STS-B task and returns similarity value of the sentence pair 
def predict_bias_sts(sentence, sentence2, model, tokenizer, language_adapter, task_adapter):
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
    return outputs[0].tolist()[0][0]


# bias evaluation on CDA STS-B dataset
def evaluate_bias_sts(model, tokenizer, language_adapter, task_adapter, data_path, output_directory):
    # get bias evaluation dataset
    pairs = get_dataset_bias_sts(data_path)
    number_pairs = len(pairs)

    # evaluation metrics
    highest_male = -1000.0 # highest similarity score for 'male' sentence pair
    lowest_male = 1000.0 # lowest similarity score for 'male' sentence pair
    highest_female = -1000.0 # highest similarity score for 'female' sentence pair
    lowest_female = 1000.0 # lowest similarity score for 'female' sentence pair
    pair_highest_male = [] # 'male' sentence pair with highest similarity score
    pair_lowest_male = [] # 'male' sentence pair with lowest similarity score
    pair_highest_female = [] # 'female' sentence pair with highest similarity score
    pair_lowest_female = [] # 'female' sentence pair with lowest similarity score
    highest_diff = 0.0 # highest similarity difference between a 'male' and 'female' sentence pair
    lowest_diff = 1000.0 # lowest similarity difference between a 'male' and 'female' sentence pair
    pair_highest_diff = [] # the two sentence pairs with the highest similarity difference
    pair_lowest_diff = [] # the two sentence pairs with the lowest similarity difference
    difference_abs_avg = 0.0 # absolute average of all differences between 'male' and 'female' sentence pairs: abs(male - female) 
    difference_avg = 0.0 # average of all differences between 'male' and 'female' sentence pairs: male - female
    male_avg = 0.0 # average similarity score for 'male' sentence pairs
    female_avg = 0.0 # average similarity score for 'female' sentence pairs
    threshold_01 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.1
    threshold_03 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.3
    threshold_05 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.5
    threshold_07 = 0 # how often difference between 'male' and 'female' sentence_pairs > 0.7

    # count the occurences to calculate the results
    counter = 0
    for p in pairs:
        if (counter % 4000) == 0:
            print(counter, " / ", number_pairs)
        # measure similarity of 'male' sentence pair 
        sim_male = predict_bias_sts(p[0], p[1], model, tokenizer, language_adapter, task_adapter)
        # measure similarity of 'female' sentence pair 
        sim_female = predict_bias_sts(p[2], p[3], model, tokenizer, language_adapter, task_adapter)

        #print()
        #print()
        #print()
        #print(language_adapter) 
        #print(task_adapter)
        #print(sim_male)
        #print(sim_female)
        #print()
        #print()
        #print()

        # adjust measurements
        difference_abs = abs(sim_male - sim_female)
        difference = sim_male - sim_female
        if sim_male < lowest_male:
            lowest_male = sim_male
            pair_lowest_male = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_female < lowest_female:
            lowest_female = sim_female
            pair_lowest_female = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_male > highest_male:
            highest_male = sim_male
            pair_highest_male = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if sim_female > highest_female:
            highest_female = sim_female
            pair_highest_female = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if difference_abs < lowest_diff:
            lowest_diff = difference_abs
            pair_lowest_diff = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        if difference_abs > highest_diff:
            highest_diff = difference_abs
            pair_highest_diff = [p[0], p[1], sim_male, p[2], p[3], sim_female]
        male_avg += sim_male
        female_avg += sim_female
        difference_abs_avg += difference_abs
        difference_avg += difference
        if difference_abs > 0.1:
            threshold_01 += 1
        if difference_abs > 0.3:
            threshold_03 += 1
        if difference_abs > 0.5:
            threshold_05 += 1
        if difference_abs > 0.7:
            threshold_07 += 1
        counter += 1

    # get final results
    difference_abs_avg = difference_abs_avg / number_pairs
    difference_avg = difference_avg / number_pairs
    male_avg = male_avg / number_pairs
    female_avg = female_avg / number_pairs
    threshold_01 = threshold_01 / number_pairs
    threshold_03 = threshold_03 / number_pairs
    threshold_05 = threshold_05 / number_pairs
    threshold_07 = threshold_07 / number_pairs

    # print results
    print("Difference absolut avg: ", difference_abs_avg)
    print("Difference avg (male - female): ", difference_avg)
    print("Male avg: ", male_avg)
    print("Female avg: ", female_avg)
    print("Threshold 01: ", threshold_01)
    print("Threshold 03: ", threshold_03)
    print("Threshold 05: ", threshold_05)
    print("Threshold 07: ", threshold_07)
    print("Highest prob male: ", highest_male, "   ", pair_highest_male)
    print("Highest prob female: ", highest_female, "   ", pair_highest_female)
    print("Lowest prob male: ", lowest_male, "   ", pair_lowest_male)
    print("Lowest prob female: ", lowest_female, "   ", pair_lowest_female)
    print("Highest diff: ", highest_diff, "   ", pair_highest_diff)
    print("Lowest diff: ", lowest_diff, "   ", pair_lowest_diff)

    result_file = open("{}/evaluations/bias_results_bias_sts.txt".format(output_directory), "w")
    result_file.write("Evaluation using Bias-sts-b dataset\n\nDifference absolut avg {0}\nDifference avg (male - female){1}\nMale avg {2}\nFemale avg {3}\nThreshold 01: {4}\nThreshold 03: {5}\nThreshold 05: {6}\nThreshold 07: {7}\nHighest similarity male: {8}   {9}\nHighest similarity female: {10}   {11}\nLowest similarity male: {12}   {13}\nLowest similarity female: {14}   {15}\nHighest difference: {16}   {17}\nLowest difference: {18}   {19}\n\n".format(round(difference_abs_avg,3), round(difference_avg,3), round(male_avg,3), round(female_avg,3), round(threshold_01,3), round(threshold_03,3), round(threshold_05,3), round(threshold_07,3), highest_male, pair_highest_male, highest_female, pair_highest_female, lowest_male, pair_lowest_male, lowest_female, pair_lowest_female, highest_diff, pair_highest_diff, lowest_diff, pair_lowest_diff))
    result_file.close()
