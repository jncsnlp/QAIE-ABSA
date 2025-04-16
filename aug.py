import os
import pprint
import google.generativeai as palm
from tqdm import tqdm
import time

palm.configure(api_key='')
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"]  = ""
# models = [
#     m
#     for m in palm.list_models()
#     if "generateText" in m.supported_generation_methods
# ]
# model = models[0].name
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
# print(model)

# data_path = './fdata-acos/rest/20/train_k_20_seed_12347.txt'
# data_aug_path = './fdata-acos/rest/20/aug.txt'
data_path = './fdata/rest15/5/train_k_5_seed_12347.txt'
data_aug_path = './fdata/rest15/5/aug.txt'

senot = {'positive': 'negative', 'negative': 'positive'}
def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    reviews,  sents, labels = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                reviews.append(words)
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")

    return sents, reviews, labels

def ask_question(question, x):

    if x == 0:
        prompt = f"Q: {question}\nA:"  # 格式化问题

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.7,
        max_output_tokens=1024,
    )

    answer = completion.result  # 从API响应中提取回答
    return answer

sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)

def at_sm(at ,review):

    question =  f'''<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{at}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: drink, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> good beverage.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>'''
    answer = ask_question(question, 0)

    return answer

def ot_sm(ot, review):
    question =  f'''<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> great drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>''' 
    answer = ask_question(question, 0)

    return answer

def ot_op(ot, review):

    question =  f'''<Input> Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a opposite word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: <Input> Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a opposite word ..., Principle: Except for the word ..., Tip: If you can't ..., <Output> bad drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    <Output>''' 
    answer = ask_question(question, 0)

    return answer

def find_label(sentence, review):
    
    question =  f'''<Input> Task Description: You can find out the difference between the given two texts by Implementation Details. Give you these elements: Original Text, Augmentation Text, Implementation Details, An input and output example, Principle, Tip. Finally output difference between the Original Text and the Augmentation Text.
                            Original Text: "{review}".
                            Augmented Text: "{sentence}".
                            Implementation Details: The text generated when the original text ''Original Review Text'' is replaced with the specified string is ''Augmented Review Text''. Find the string used to replace the specified string.
                            An input and output example: <Input> Task Description: You can find ..., Original Text: good drink, Replaced Word: good beverage, Implementation Details: The text generated..., Principle: Except for the word.., Tip: If you can't..., <Output> beverage
                            Principle: The output string must be found in the text ''Augmented Review Text''.
                            Tip: If you can't find this label, you can gradually narrow it down by elimination. You can use context clues to infer the possible string.
                    <Output>'''
    answer = ask_question(question, 0)

    return answer



text = []
for i, (review, label) in enumerate(zip(reviews, labels)):
    if len(label) == 1:
        for quad in label:
            at, ac, sp, ot = quad
            text.append(review + "####" + str([(at, ac, sp, ot)]))

            if at != 'none':
                ats_sentence = at_sm(at, review)
                new_at = find_label(ats_sentence, review)
                if ats_sentence is not None and new_at in ats_sentence:
                    text.append(ats_sentence + "####" + str([(new_at, ac, sp, ot)]))

                if ot != 'none':
                    ots_sentence = ot_sm(ot, review)
                    new_ots = find_label(ots_sentence, review)
                    if ots_sentence is not None and new_ots in ots_sentence:
                        text.append(ots_sentence + "####" + str([(at, ac, sp, new_ots)]))

                    at_ots_sentence = ats_sentence.replace(ot, new_ots)
                    if at_ots_sentence is not None and new_at in at_ots_sentence and new_ots in at_ots_sentence:
                        text.append(at_ots_sentence + "####" + str([(new_at, ac, sp, new_ots)]))

                    if sp == 'positive' or sp == 'negative':
                        oto_sentence = ot_op(ot, review)
                        new_oto = find_label(oto_sentence, review)

                        if oto_sentence is not None and new_oto in oto_sentence:
                            text.append(oto_sentence + "####" + str([(at, ac, senot[sp], new_oto)]))

                        at_oto_sentence = ats_sentence.replace(ot, new_oto)
                        if at_oto_sentence is not None and new_oto in at_oto_sentence:
                            text.append(at_oto_sentence + "####" + str([(new_at, ac, senot[sp], new_oto)]))

            elif at == 'none':
                if ot != 'none':
                    ots_sentence = ot_sm(ot, review)
                    new_ots = find_label(ots_sentence, review)
                    if new_ots == None:
                        break
                    if ots_sentence is not None and new_ots in ots_sentence:
                        text.append(ots_sentence + "####" + str([(at, ac, sp, new_ots)]))

                    if sp == 'positive' or sp == 'negative':
                        oto_sentence = ot_op(ot, review)
                        new_oto = find_label(oto_sentence, review)
                        if new_oto == None :
                            break
                        if oto_sentence is not None and new_oto in oto_sentence:
                            text.append(oto_sentence + "####" + str([(at, ac, senot[sp], new_oto)]))


    else:
        label_new = []
        for qua in label:
            label_new.append(list(qua))

        text.append(review + "####" + str(label))
        at, ac, sp, ot = label_new[0]
        senten = review #初始化
        label11_lis = label_new #初始化
        if at != 'none':
            ats_sentence = at_sm(at, review)
            new_at = find_label(ats_sentence, review)

            label11 = []
            label11_lis = []
            for qu in label_new:
                my_list = []
                if qu[0] == at:
                    my_list = [new_at, qu[1], qu[2], qu[3]]
                else:
                    my_list = qu
                label11.append(tuple(my_list))
                label11_lis.append(my_list)

            senten = ats_sentence
            if ats_sentence is not None and new_at in ats_sentence:
                text.append(ats_sentence + "####" + str(label11))

            if ot != 'none':
                at_ots_sentence = ot_sm(ot, ats_sentence)
                new_at_ots = find_label(at_ots_sentence, ats_sentence)

                label12 = []
                label12_lis = []
                for qu in label11_lis:
                    my_list = []
                    if qu[3] == ot:
                        my_list = [qu[0], qu[1], qu[2], new_at_ots]
                    else:
                        my_list = qu
                    label12.append(tuple(my_list))
                    label12_lis.append(my_list)

                label11_lis = label12_lis
                senten = at_ots_sentence
                if at_ots_sentence is not None and new_at_ots in at_ots_sentence:
                    text.append(at_ots_sentence + "####" + str(label12))

        elif at == 'none':
            if ot != 'none':
                ots_sentence = ot_sm(ot, review)
                new_ots = find_label(ots_sentence, review)

                label13 = []
                label11_lis = []
                for qu in label_new:
                    my_list = []
                    if qu[3] == ot:
                        my_list = [qu[0], qu[1], qu[2], new_ots]
                    else:
                        my_list = qu

                    label13.append(tuple(my_list))
                    label11_lis.append(my_list)
                senten = ots_sentence
                if ots_sentence is not None and new_ots in ots_sentence:
                    text.append(ots_sentence + "####" + str(label13))


        at, ac, sp, ot = label_new[1]

        if at != 'none' and at != label_new[0][0]:
            ats_sentence = at_sm(at, senten)
            new_at = find_label(ats_sentence, senten)

            label21 = []
            label21_lis = []
            for i in range(len(label11_lis)):
                my_list = []
                if i == 0:
                    my_list = label11_lis[i]
                else:
                    if label11_lis[i][0] == at:
                        my_list = [new_at, label11_lis[i][1], label11_lis[i][2], label11_lis[i][3]]
                    else:
                        my_list = label11_lis[i]

                label21.append(tuple(my_list))
                label21_lis.append(my_list)
            if ats_sentence is not None and new_at in ats_sentence:
                text.append(ats_sentence + "####" + str(label21))

            if ot != 'none' and ot != label_new[0][3]:
                at_ots_sentence = ot_sm(ot, ats_sentence)
                new_at_ots = find_label(at_ots_sentence, ats_sentence)

                label22 = []
                for i in range(len(label21_lis)):
                    my_list = []
                    if i == 0:
                        my_list = label21_lis[i]
                    else:
                        if label21_lis[i][3] == ot:
                            my_list = [label21_lis[i][0], label21_lis[i][1], label21_lis[i][2], new_at_ots]
                        else:
                            my_list = label21_lis[i]

                    label22.append(tuple(my_list))
                if at_ots_sentence is not None and new_at_ots in at_ots_sentence:
                    text.append(at_ots_sentence + "####" + str(label22))

        elif at == 'none':
            if ot != 'none' and ot != label_new[0][3]:
                ots_sentence = ot_sm(ot, senten)
                new_ots = find_label(ots_sentence, review)

                label23 = []
                for i in range(len(label11_lis)):
                    my_list = []
                    if i == 0:
                        my_list = label11_lis[i]
                    else:
                        if label11_lis[i][3] == ot:
                            my_list = [label11_lis[i][0], label11_lis[i][1], label11_lis[i][2], new_ots]
                        else:
                            my_list = label11_lis[i]

                    label23.append(tuple(my_list))
                if ots_sentence is not None and new_ots in ots_sentence:
                    text.append(ots_sentence + "####" +str(label23))

with open(data_aug_path, 'w') as f:
    for item in text:
        f.write("%s\n" % item)


