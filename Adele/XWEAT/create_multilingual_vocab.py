import csv

languages = ['vocab_en_de.csv', 'vocab_en_es.csv', 'vocab_en_hr.csv', 'vocab_en_it.csv', 'vocab_en_ru.csv', 'vocab_en_tr.csv']

multilingual_vocab = []

# extract english vocabulary
with open('./data/vocab_en.txt', 'r', encoding='utf-8') as english_vocab:
    print("vocab_en.txt")
    lines = english_vocab.read().split('\n')
    for word in lines:
        multilingual_vocab.append(word.lower())

# extract vocabulary for each language in languages
for language in languages:
    print(language)
    with open('./data/' + language, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for word in row[1:]:
                if word != "" and word is not None:
                    multilingual_vocab.append(word.lower())

# save vicabulary
with open('./data/multilingual_vocab.txt', 'w', encoding='utf-8') as txt_file:
    for word in multilingual_vocab:
        txt_file.write("%s\n" % word)