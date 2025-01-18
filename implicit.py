import os
import pprint
import google.generativeai as palm
from tqdm import tqdm

palm.configure(api_key='AIzaSyAxtaIdclFbgAbuWfOjWjgFN3aBki3dKhQ')
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"]  = "http://127.0.0.1:7890"
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

# data_path = './fdata-acos/rest/50/dev_k_50_seed_12347.txt'
# datai_path = './fdata-acos/rest/50/dev_im.txt'

# data_path = './fdata/lap14/5/aug.txt'
# datai_path = 'fdata/lap14/5/aug_im.txt'

data_path = './fdata/lap14/10/train_k_10_seed_12347.txt'
datai_path = './fdata/lap14/10/train_im.txt'

# data_path = './data/lap14/test.txt'
# datai_path = './data/lap14/test_im.txt'

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
    # print(sents[:2])
    # print(reviews[:2])
    # print(labels[:2])
    return sents, reviews, labels

def ask_question(question, x):
    # prompt = "Who are you?"
    prompt = "What does the review ‘you ca n ' t go wrong here . ' imply? "
    if x == 0:
        prompt = f"Q: {question}\nA:"  # 格式化问题
    if x == 1:
        prompt = f"Q: {question1}\nA: {answer1}\nQ: {question}\nA:"
    if x == 2:
        prompt = f"Q: {question1}\nA: {answer1}\nQ: {question2}\nA: {answer2}\nQ: {question}\nA:"
        # print(prompt)

    # answer briefly
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.7,
        # The maximum length of the response
        max_output_tokens=1024,
    )

    answer = completion.result  # 从API响应中提取回答
    return answer

sents, reviews, labels = read_line_examples_from_file(data_path, silence=True)



answers = []
i = 0

for review in reviews:

    question1 = f'''Given the text "{reviews[i]}", what is your understanding of the text? Keep your answers short.'''
    answer1 = ask_question(question1, 0)
    question2 = f'''The text "{reviews[i]}" is a comment from the restaurant field. answer briefly. "{answer1}" What does the sentence imply ? answer briefly.'''
    answer2 = ask_question(question2, 0)
    answers.append(answer2)
    print(answer2)
    i = i + 1
    # if i == 8:
    #     break

with open(datai_path, 'w') as f:
    for item in answers:
        f.write("%s\n" % item)