import os
import ast
import json
import pandas as pd 
from span_marker import SpanMarkerModel

# Download from the 🤗 Hub
model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-roberta-large-ontonotes5")

import re
def normalizer(phrase):
    regex = r"([\s,*+:!?\\-\]\[])"
    split_phrase = re.split(regex, phrase.lower())
    return list(filter(None, split_phrase))

import openai
openai.api_key = "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"
system_message = [{"role": "system", "content": """You are an AI labeler assistant. I am building a Name Entity Recognition to help a speech engine pronounce words and numbers as they should be spoken. I need you to provide the appropriate label for each token in the array (ignore whitespaces), use the utterance to give additional context. Decide if the span should be labeled as one of the following classes (given with their definition): 

- characters (The text is spoken as individual letters and spoken as if its spelled out, for example x ,  T E S T)
- cardinal (Any number that refers to a quantity that should be spoken differently then written such as two hundred thousand, nine hundred and fifty, thirty five, etc..., It does not have fractions but may include decimals, but do not include anything is a date)
- ordinal (if a number or text indicates the order of something and should be spoken as first, second, third, etc...) 
- number_digit (Individual digits listed such that if spoken it would be spoken as a sequence of individual digits. This only applies to numbers that are integers)
- unit (A number that is qualified by a unit, for example 10 feet, 6 foot, 1 cup, 1.5 ms, etc...) 
- fraction (A fractional value such as 3/8 of an inch) 
- date (Any valid representation of a date which includes a reference to a year, month, date. For example 10/23/22, October 23rd)
- time (Anything that refers to the reporting the the time on a clock, for example 4:00am, or four am)
- duration (Any text referring to how much time elapsed or how long something should take)
- telephone (Something that should be interpreted as a telephone number, for example (888) 555-1212 or 888-555-1212, etc...) 
- address (Any text that includes references to an address such as 150th CT NE, Redmond, WA)
- currency (Any text that should be spoken as a currency for example $9.99)
- location (Any text the references a place such as a city, country, etc...)
- name (Any text that refers to the name of a person, event, organization, etc...)

Rules:
- The list is ordered from least to most specific.
- Always choose the most relevant and specific class label from the list. 
- Always return valid JSON with the fields labels, and optional explanation field. 
- The length of the labels field must equal the length of the tokens field.
- In cases where you have low confidence, provide a brief reasoning in the explanation field. 
                                               """}, 
{"role": "user", 
"content": '{ "tokens": ["question",  ":", "put",  "1933/43", ",", "-0.06", ",", "-5", ",", "2", "in","ascending","order.","answer",":", "-5", ",", "-0.06", ",", "2", ",", "1933/43"] }'
},
{
    "role": "assistant",
    "content": '{ "labels": [("question", "None"), (":", "None"), ("put", "None"), ("1933/43, "fraction"), (",", "None"), ("-0.06", "cardinal"), (",", "None"), ("2", "cardinal"), ("in", "None"), ("ascending", "None"),  ("order.", "None"), ("answer", "None"), (":", "None"), ("-5", "cardinal"), (",", "None"), ("-0.06", "cardinal"), (",", "None"), ("2", "cardinal"), (",", "None"), ("1933/43, "fraction")] }'
}]


qa_pairs = []
for root, dirs, files in os.walk(".", topdown=False):
   for name in files:
    if name.endswith('.txt'):
        print(os.path.join(root, name))
        with open(os.path.join(root, name)) as f:
            q = "None"
            a = "None"
            for i,line in enumerate(f):
                if i%2 == 0:
                    q = line.strip()
                else:
                    a = line.strip()
                    utterance = f'question: {q} answer: {a}'
                    messages = list(system_message)
                    normalized_utt = normalizer(utterance)
                    filtered_down = list(filter(lambda x: x is not None and x != "None" and x != '' and x!= ' ', normalized_utt))
                    messages.append({
                        "role": "user",
                        "content": f"\"tokens\": [{', '.join(filtered_down)}]"
                    })
                    chat_completion = openai.ChatCompletion.create(model="gpt-4", 
                                                                    messages=messages)
                    if chat_completion and chat_completion['choices']:
                        print(filtered_down)
                        print(chat_completion)
                        try:
                            gpt_predicted_label = ast.literal_eval(chat_completion['choices'][0]['message']['content'])
                            print(len(gpt_predicted_label['labels']), len(filtered_down))
                            qa_pairs.append({"utterance": utterance, 'filtered_tokens': filtered_down, 'normalized_tokens': normalized_utt, 'label_predictions': gpt_predicted_label['labels'] })
                        except Exception as e:
                            print("Skipping ", utterance)
                            print(e)
                            print()
                    q = "None"
df = pd.DataFrame(qa_pairs)
df.to_csv('math_phrases.csv')