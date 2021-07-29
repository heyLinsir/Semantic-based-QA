import csv
import sys

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

data_type = sys.argv[1]

def read_csv(input_file):
    """Reads a csv file."""
    lines = []
    with open(input_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            lines.append(row)
    return lines

def filter_answer(answer):
    if answer.lower().startswith('none of the above'):
        return False
    if answer == "":
        return False
    return True

def convert():
    records = read_csv('./%s.csv' % (data_type))
    examples = []
    for (i, record) in enumerate(records):
        guid = record["id"]
        context = record["context"]
        question = record["question"]
        label = int(record["label"])

        new_question = convert_question(question)
        if new_question is not None:
           question = new_question

        if not filter_answer(record["answer%d" % (label)]):
            continue
        for j in range(4):
            if not filter_answer(record["answer%d" % (j)]):
                for k in range(4):
                    if k == label or k == j:
                        continue
                    if filter_answer(record["answer%d" % (k)]):
                        record["answer%d" % (j)] = record["answer%d" % (k)]
                        break
        examples.append("%s\t%s\t%s\t%s\t%s\t%s\t%d" % (context, question, 
            record["answer0"], record["answer1"], record["answer2"], record["answer3"], 
            label))
    print(len(examples))

    return examples

def lemma(sentence):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    tokens = sentence.split(' ')
    try:
        tagged_sent = pos_tag(tokens)
    except:
        return None, None

    wnl = WordNetLemmatizer()
    lemmas_sent, tags = [], []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        tags.append(wordnet_pos)
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return ' '.join(lemmas_sent), tags

def convert_question(question):
    conj_words = ['if', 'during', 'because', 'despite', 'when', 'while', 'after', 'once']
    pre_question = ''
    question = question.strip()
    question = question[0].lower() + question[1:]
    question_words = question.split(' ')[:-1]
    for word in conj_words:
        if word in question_words[2:]:
            word_index = question_words.index(word)
            pre_question = ' '.join(question_words[word_index:] + [','])
            question_words = question_words[:word_index]
            break

    question = ' '.join(question_words)
    lemma_question, tags = lemma(question)
    if lemma_question is None:
        return None

    if question_words[0] == 'what':
        if len(lemma_question.split(' ')) <= 4 and 'happen' in lemma_question:
            # new_question = ''
            new_question = ' '.join(['it'] + question_words[1:] + ['that'])
        elif len(lemma_question.split(' ')) == 4 and ' '.join(lemma_question.split(' ')[:2] + [lemma_question.split(' ')[3]]) == 'what do do':
            new_question = question_words[2]
            if new_question == 'you':
                new_question = 'I'
        elif 'a possible reason' in ' '.join(question_words[:6]):
            new_question = ' '.join(question_words[question_words.index('reason') + 1:] + ['because'])
        elif tags[1] == 'n' and lemma_question.split(' ')[1] not in ['may', 'have', 'do', 'be', 'will', 'can']:
            new_question = ' '.join(['the'] + question_words[1:] + ['is', 'that'])
        else:
            new_question = ' '.join(['it'] + question_words[1:] + ['that'])
    elif question_words[0] == 'why':
        new_question = ' '.join(question_words[2:] + ['because'])
    else:
        return None

    if new_question == '':
        new_question = pre_question
    elif pre_question != '':
        new_question = pre_question + ' ' + new_question
    new_question = new_question[0].upper() + new_question[1:]
    return new_question


with open('CosmosQA-PQA.%s' % (data_type), 'w') as w:
    w.write('\n'.join(convert()))

