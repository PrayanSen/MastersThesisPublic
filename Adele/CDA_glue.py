import csv
import argparse


# checks if list already contains the word pair
def is_pair_in_list(all_pairs, pair):
    for p in all_pairs:
        if (p[0] == pair[0]) and p[1] == pair[1]:
            return True
    return False


# returns word list of noun pairs of Zhao et al. and 100 self-created name pairs
def get_gender_word_pairs():
    word_pairs = []

    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/generalized_swaps.txt
    # creates list with word pairs --> [ [pair1[0], pair1[1]] , [pair2[0], pair2[1]] , ... ]
    file_wordlist = open('datasets/wordpairs/cda_word_pairs_gender.txt', 'r') 
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        word_pairs.append(word_pair)

    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/extra_gendered_words.txt
    # appends additional word pairs from extra file
    file_wordlist = open('datasets/wordpairs/cda_word_pairs_gender_extra.txt', 'r') 
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_pairs, word_pair):
            word_pairs.append(word_pair)
            word_pairs.append([word_pair[1], word_pair[0]]) # both 'dircetions' needed: (male, female) and (female, male)
        
    # https://www.ssa.gov/oact/babynames/limits.html
    # gets the top 100 names of 2019 for boys and girls and appends the pairs (male, female) and (female, male) to the word pair list
    file_wordlist = open('datasets/wordpairs/cda_word_pairs_names.txt', 'r') 
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_pairs, word_pair):
            word_pairs.append(word_pair)

    return word_pairs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="The dataset that should be counterfactual augmented.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="The output file for the counterfactual augmented dataset.")
    parser.add_argument("--cda_type",
                        type=str,
                        required=True,
                        help="1-sided or 2-sided")
    parser.add_argument("--dataset_type",
                        type=str,
                        required=True,
                        help="Can be train, dev or test. I only used augmented train files! dev and test stay original to test actual performance.")
    parser.add_argument("--task",
                        type=str,
                        required=True,
                        help="mnli or sts-b")
    args = parser.parse_args()

    print("Get gender word pairs...")
    word_pairs = get_gender_word_pairs()
    print("...done\n\n")

    # Augment a sentence pair if a word of word_list is in at least one of the sentences
    print("CDA of dataset...")
    new_file = []
    tsv_file = open(args.input_file, 'r', encoding='utf-8') 
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    i = 0
    for row in read_tsv:
        if i == 0: # header row
            i += 1
            new_file.append(row)
            continue
        i += 1
        edit = False
        # lower the two sentences
        if args.task == "mnli":
            row[8] = row[8].lower()
            row[9] = row[9].lower()
        else: # sts-b
            row[7] = row[7].lower()
            row[8] = row[8].lower()
        # if 2-sided cda: append unchanged row to new file
        if args.cda_type == "2-sided":
            new_file.append(row)
        # get sentences of MNLI or STS-B dataset and split them to words
        if args.task == "mnli":
            s1 = row[8].split() # words of first sentence
            s2 = row[9].split() # words of second sentence
        else: # sts-b
            s1 = row[7].split() # words of first sentence
            s2 = row[8].split() # words of second sentence
        # check for each word of both sentences if it is in word list
        for j in range(len(s1)):
            for word_pair in word_pairs:
                # if there is a match, switch word with corresponding partner word and set edit to True
                if s1[j] == word_pair[0]:
                    s1[j] = word_pair[1]
                    edit = True
                    break
        for j in range(len(s2)):
            for word_pair in word_pairs:
                # if there is a match, switch word with corresponding partner word and set edit to True
                if s2[j] == word_pair[0]:
                    s2[j] = word_pair[1]
                    edit = True
                    break
                
        # if one of the sentences contains a word of word list, edit is True and the augmented sentence pair is added to new dataset
        if edit:
            if args.task == "mnli":
                new_row = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]," ".join(s1)," ".join(s2)]
                if args.dataset_type != "test": # train and dev dataset have label
                    for col in row[10:]:
                        new_row.append(col)
            else: # sts-b
                new_row = [row[0],row[1],row[2],row[3],row[4],row[5],row[6]," ".join(s1)," ".join(s2)]
                if args.dataset_type != "test": # train and dev dataset have label
                    new_row.append(row[9])

            new_file.append(new_row)
    print("...done")

    print("Create new file and write sentences to it...")
    with open(args.output_file, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(new_file)
    print("...done")
