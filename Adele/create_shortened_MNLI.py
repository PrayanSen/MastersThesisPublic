import random
import csv
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Path to MNLI train file.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="The output for the shorten MNLI dataset.")
    parser.add_argument("--size",
                        type=int,
                        required=True,
                        help="Number of instances for new shortened MNLI dataset.")
    args = parser.parse_args()


    # open MNLI file and get rows
    tsv_file = open(args.input_file, 'r', encoding='utf-8') 
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    print("Get old sentences...")
    original_dataset = []
    for row in read_tsv:
        original_dataset.append(row)
    print("...done")
    print("Number of old sentences: ", len(original_dataset), "\n")


    print("Get random sentences...")
    shortened_dataset = []
    entailment = 0    # counts number of already added instances that have entailment label
    neutral = 0       # counts number of already added instances that have neutral label
    contradiction = 0 # counts number of already added instances that have contradiction label
    duplicates = []
    i = 0
    # adds instances to new MNLI file until desired size is reached
    while i < args.size:
        # creates random index and adds the sentence from original MNLI file to shortened new file if it is not a duplicate and label ditribution can still be balanced in the end
        rand = random.randint(1,len(original_dataset)-1)
        if rand not in duplicates:
            if original_dataset[rand][-1] == "entailment":
                entailment += 1
                if entailment >= (args.size / 3) + 1: # checks if label distribution can still be balanced in the end
                    entailment -= 1
                    continue
            if original_dataset[rand][-1] == "neutral": # checks if label distribution can still be balanced in the end
                neutral += 1
                if neutral >= (args.size / 3) + 1:
                    neutral -= 1
                    continue
            if original_dataset[rand][-1] == "contradiction": # checks if label distribution can still be balanced in the end
                contradiction += 1
                if contradiction >= (args.size / 3) + 1:
                    contradiction -= 1
                    continue

            # if instance is not a duplicate and label distribution is fine, add it to list of duplicates and add it to new shortened file
            duplicates.append(i)
            i += 1  
            shortened_dataset.append(original_dataset[rand])
    print("...done")
    print("Number of new sentences: ", len(shortened_dataset), "\n")


    print("Create new file and write instances of shortened MNLI to it...")
    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(shortened_dataset)
    print("...done")