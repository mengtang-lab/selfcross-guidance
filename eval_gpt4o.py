import argparse
import os
import re
import yaml
import spacy
import base64
import openai
import json

API_KEY = "your openai api key"
client = openai.OpenAI(api_key=API_KEY)


def prompt_factory(PROMPT, classes):
    prompt = f"""
You are now an expert to check the difference of objects in the synthesized images. The prompt is f"{PROMPT}".
Based on the image description below, reason and answer the following questions:
Image Description: "{PROMPT}"
1) Are the generated {classes} recognizable and regular (without artifacts) in terms of its shape and semantic structure? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
2) Are the generated {classes} have different appearance? Give a True/False answer after reasoning. Please ignore differences in size, Body proportions and poses.
"""
    return prompt
# 1) Are there {number} {classes} appearing in this image? Give a True/False answer after reasoning.

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def eval_gpt4o(image_path, number, classes):
    base64_image = encode_image(image_path)

    prompt = prompt_factory(number, classes)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                    },
                },
            ],
        }
        ],
        max_tokens=300,
    )
    answers = response.choices[0].message.content
    answers = re.sub(r'\n\s*\n', '\n', answers)
    answers = answers.lower()
    # print(f"{answers}\n")
    return answers


def decode_answers(answers, nlp):
    results = {}

    question_keys = {
            # 1: "counting",
            1: "recognizable",
            2: "diversity"
        }

    # current_question = None
    current_question_key = None
    
    # Split the text into lines and process each line
    lines = answers.splitlines()

    combined_lines = []
    for i, line in enumerate(lines):
        print(i,line)
        line = line.strip()
        if line and line[0].isdigit() and ')' in line:
            combined_lines.append(line)
        elif len(combined_lines)==0: continue # remove irrelevant beginning words
        else:
            combined_lines[-1]=combined_lines[-1]+" "+line
    # print(len(combined_lines),combined_lines)

    for line in combined_lines:
        line = line.strip()
        doc = nlp(line.lower())
        contain_true = any(token.text == 'true' for token in doc) #or any(token.text == 'yes' for token in doc)
        contain_false = any(token.text == 'false' for token in doc) #or any(token.text == 'no' for token in doc) or any(token.text == 'n/a' for token in doc)
        
        # get the question number
        if line and line[0].isdigit() and ')' in line:
            current_question = int(line[0])
            
            if current_question in question_keys:
                current_question_key = question_keys[current_question]
            else:
                current_question_key = None

        # print(i, line, current_question_key, contain_true, contain_false)
        
        # Process lines that could be answers or reasoning
        if current_question_key:
            if contain_true and not contain_false:
                results[current_question_key] = True
                current_question_key = None  # Reset after finding an answer
            
            elif contain_false and not contain_true:
                results[current_question_key] = False
                current_question_key = None  # Reset after finding an answer

            elif contain_true and contain_false:
                answer_tokens = [token.text for token in doc if token.text in ['true', 'false']]
                # if answer_tokens == ['true', 'false']:
                #     continue
                results[current_question_key] = (answer_tokens[-1] == 'true')
                current_question_key = None  # Reset after finding an answer
            else:
                results[current_question_key] = None
                current_question_key = None

    return results


def decode_dir(dir_name):
    match = re.match(r'a photo of (\w+) (\w+)', dir_name)

    if match:
        number = match.group(1)
        classes = match.group(2)
        return number, classes
    else:
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CoCoCount', type=str, help="Dataset type")
    parser.add_argument("--image_root", default="./SD3/outputs", type=str, help="Path to the image file")
    parser.add_argument("--output_dir", default="./SD3", type=str, help="Path to the output directory")
    parser.add_argument("--resume", default=False, type=str, help="Resume from the last checkpoint")
    parser.add_argument("--prompt_eng", default=False, type=str, help="Resume from the last checkpoint")

    args = parser.parse_args()
    
    # iterate directories
    root = args.image_root
    output_dir = args.output_dir
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    out_name = os.path.basename(root).split('(')[0]

    nlp = spacy.load("en_core_web_sm")

    # if resume, read from old yaml file
    if args.resume:
        with open(os.path.join(output_dir, f"{out_name}.yaml"), 'r') as f:
            done_list = yaml.safe_load(f)
        done_keys = []
        done_keys = set(done_list.keys())

    with open('./data/CoCoCount.json') as f:
        data = f.read()
    print("Data type before reconstruction : ", type(data))
    # reconstructing the data as a dictionary
    data = json.loads(data)
    print("Data type after reconstruction : ", type(data))


    with open(os.path.join(output_dir, f"{out_name}{args.resume}.yaml"), 'w') as f:

        for i in range(len(data)):
            PROMPT = data[i]['prompt']
            SEED = data[i]['seed']
            number = data[i]['int_number']
            SEED = int(SEED)



            if args.prompt_eng:
                new_string = PROMPT.replace(",", " ")
                words = new_string.split(" ")  # tokenizer treats comma as an individual token

                if isinstance(data[i]['object_plural'], list):  # for T2I-comp-bench
                    keywords = data[i]['object_plural']  # "could be single word like "gloves" only!"
                else:  # for coCoCount
                    keywords = data[i]['object_plural'].split(" ")  # "could be multiple words like "baseball gloves" "
                # print("keywords : ", keywords, "words : ",words)

                new_list = []
                for word in words:
                    new_list.append(word)
                    if word == keywords[-1]:
                        new_list.append('with different appearance')
                # print(new_list)
                words = new_list
                PROMPT = " ".join(words)

            image_url = root + f"/{PROMPT}-{SEED}.png"
            classes = data[i]['object_plural']

            # if number != 7 or classes !="birds" or SEED != 847010: continue

            if not os.path.exists(image_url): continue

            if args.resume and f"{number}_{classes}_{SEED}" in done_keys:
                continue
            save_key = f"{number}_{classes}_{SEED}"
            print(save_key)
            answers = eval_gpt4o(image_url, PROMPT, classes)

            results = decode_answers(answers, nlp)
            # save results
            yaml_data = {save_key: results}
            yaml.dump(yaml_data, f)

