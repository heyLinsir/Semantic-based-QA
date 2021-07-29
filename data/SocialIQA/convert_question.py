import json
import sys
import re

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

data_type = sys.argv[1]

class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass

class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, and choices.
    """
    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
              "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
              "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
              "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
              "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
              "What will (.*) do next?": r"[SUBJ] then",
              "How would (.*) feel after?": r"[SUBJ] then",
              "How would you describe (.*)?": r"[SUBJ] is seen as",
              "What kind of person is (.*)?": r"[SUBJ] is seen as",
              "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
              "Why did (.*) do that?": r"Before, [SUBJ] wanted",
              "Why did (.*) do this?": r"Before, [SUBJ] wanted",
              "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
              "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What will happen to (.*)?": r"[SUBJ] then",
              "What will happen to (.*) next?": r"[SUBJ] then"
        }

    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        return context, question, choices

    def fields_to_instance(self, fields):
        context, question, choices = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
                break

        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")

        choices = [
            " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
                "?", "").replace("wanted to wanted to", "wanted to").replace(
                "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

        answer_prefix = answer_prefix.replace("?", "")
        new_choices = [choice[len(answer_prefix):].strip() for choice in choices]

        context_with_choices = [f"{context} {choice}" for choice in choices]
        return context, question, choices, context_with_choices, answer_prefix, new_choices

def convert():
    reader = SocialIQAInstanceReader()

    new_items = []
    with open(data_type + '.jsonl', encoding="utf-8") as f:

        for text in f:

            item = json.loads(text)

            context, question, choices, context_with_choices, answer_prefix, new_choices = reader.fields_to_instance(item)
            item['context_with_choices'] = context_with_choices
            item['question'] = answer_prefix
            item['answerA'] = new_choices[0]
            item['answerB'] = new_choices[1]
            item['answerC'] = new_choices[2]

            for choice, context_with_choice in zip(new_choices, context_with_choices):
                assert ' '.join((item['context'], item['question'], choice)) == context_with_choice

            new_items.append(json.dumps(item))

    print('Total Number:', len(new_items))

    return new_items
    

with open(data_type + '.jsonl.convertQ', 'w') as w:
    w.write('\n'.join(convert()))

