import argparse
import os
import re
import yaml
import spacy
import base64
import openai


API_KEY = "TODO_API_KEY" 
client = openai.OpenAI(api_key=API_KEY)


def prompt_factory(class_A, class_B, config='animal-animal'):
    if config == 'animal-animal' or config == 'similar-subjects':
        prompt = f"""
You are now an expert to check the faithfulness of the synthesized images. The prompt is f"a {class_A} and a {class_B}".
Based on the image description below, reason and answer the following questions:
Image Description: "a {class_A} and a {class_B}"
1) Is there {class_A} appearing in this image? Give a True/False answer after reasoning.
2) Is the generated {class_A} recognizable and regular (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
3) Is there {class_B} appearing in this image? Give a True/False answer after reasoning.
4) Is the generated {class_B} recognizable and regular (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
5) Is the generated content a mixture of {class_A} and {class_B}? An example of mixture is that Sphinx resembles a mixture of a person and a lion. Give a True/False answer after reasoning.
"""
    elif config == 'animal-object':
        if '_' in class_B:
            attribute, category = class_B.split('_')
            prompt = f"""
You are now an expert to check the faithfulness of the synthesized images. The prompt is f"a {class_A} and a {attribute} {category}".
Based on the image description below, reason and answer the following questions:
1) Is there {class_A} appearing in this image? Give a True/False answer after reasoning.
2) Is the generated {class_A} recognizable and regular (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
3) Is there {category} appearing in this image? Give a True/False answer after reasoning.
4) Is the generated {category} recognizable and regular (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
5) Is the generated {category} {attribute} in color? Ignore its shape or semantic structure. Give a True/False answer after reasoning.
6) Is the generated content a mixture of {class_A} and {category}? An example of mixture is that Sphinx resembles a mixture of a person and a lion. Give a True/False answer after reasoning.
"""
        else:
            prompt = f"""
You are now an expert to check the faithfulness of the synthesized images. The prompt is f"a {class_A} with a {class_B}".
Based on the image description below, reason and answer the following questions:
1) Is there {class_A} appearing in this image? Give a True/False answer after reasoning.
2) Is the generated {class_A} recognizable (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
3) Is there {class_B} appearing in this image? Give a True/False answer after reasoning.
4) Is the generated {class_B} recognizable (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
5) Is the generated {class_A} wearing a {class_B}? Ignore the color, style, and structure of both {class_A} and {class_B}. Give a True/False answer after reasoning.
6) Is the generated content a mixture of {class_A} and {class_B}? An example of mixture is that Sphinx resembles a mixture of a person and a lion. Give a True/False answer after reasoning.
"""
    elif config == 'object-object':
        attribute1, category1 = class_A.split('_')
        attribute2, category2 = class_B.split('_')
        prompt = f"""
You are now an expert to check the faithfulness of the synthesized images. The prompt is f"a {attribute1} {category1} and a {attribute2} {category2}".
Based on the image description below, reason and answer the following questions:
1) Is there {category1} appearing in this image? Give a True/False answer after reasoning.
2) Is the generated {category1} recognizable (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
3) Is the generated {category1} {attribute1} in color? Ignore its shape or semantic structure. Give a True/False answer after reasoning.
3) Is there {category2} appearing in this image? Give a True/False answer after reasoning.
5) Is the generated {category2} recognizable (without artifacts) in terms of its shape and semantic structure only? For example, answer False if a two-leg animal has three or more legs, or a two-eye animal has four eyes, or a two-ear animal has one or three ears. Ignore style, object size in comparison to its surroundings. Give a True/False answer after reasoning.
6) Is the generated {category2} {attribute2} in color? Ignore its shape or semantic structure. Give a True/False answer after reasoning.
7) Is the generated content a mixture of {category1} and {category2}? An example of mixture is that Sphinx resembles a mixture of a person and a lion. Give a True/False answer after reasoning.
"""
    return prompt


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def eval_gpt4o(image_path, class_A, class_B, config='animal-animal'):
    base64_image = encode_image(image_path)

    prompt = prompt_factory(class_A, class_B, config=config)

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
    print(f"{answers}\n")
    return answers


def decode_answers(answers, nlp, config='animal-animal'):
    results = {}
    if config == 'animal-animal' or 'similar-subjects':
        question_keys = {
            1: "A_appear",
            2: "A_recognizable",
            3: "B_appear",
            4: "B_recognizable",
            5: "content_mixture"
        }
    elif config == 'animal-object':
        question_keys = {
        1: "A_appear",
        2: "A_recognizable",
        3: "B_appear",
        4: "B_recognizable",
        5: "attribute",
        6: "content_mixture"
    }
    elif config == 'object-object':
        question_keys = {
        1: "A_appear",
        2: "A_recognizable",
        3: "A_attribute",
        4: "B_appear",
        5: "B_recognizable",
        6: "B_attribute",
        7: "content_mixture"
    }
    
    current_question = None
    current_question_key = None
    
    # Split the text into lines and process each line
    lines = answers.splitlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        doc = nlp(line.lower())
        contain_true = any(token.text == 'true' for token in doc) or any(token.text == 'yes' for token in doc)
        contain_false = any(token.text == 'false' for token in doc) or any(token.text == 'no' for token in doc) or \
            any(token.text == 'n/a' for token in doc)
        
        # get the question number
        if line and line[0].isdigit() and ')' in line:
            current_question = int(line[0])
            
            if current_question in question_keys:
                current_question_key = question_keys[current_question]
            else:
                current_question_key = None
        
        # Process lines that could be answers or reasoning
        if current_question_key:
            if contain_true and not contain_false:
                results[current_question_key] = True
                current_question_key = None  # Reset after finding an answer
            
            elif contain_false and not contain_true:
                results[current_question_key] = False
                current_question_key = None  # Reset after finding an answer

            elif contain_true and contain_false:
                answer_tokens = [token.text for token in doc if token.text in ['true', 'false', 'yes', 'no']]
                if answer_tokens == ['true', 'false']:
                    continue
                results[current_question_key] = (answer_tokens[-1] == 'true' or answer_tokens[-1] == 'yes')
                current_question_key = None  # Reset after finding an answer

    return results


def decode_dir(dir_name, config='animal-animal'):
    if config == 'animal-animal' or config == 'similar-subjects':
        match = re.match(r'a (\w+) and a (\w+)', dir_name)
    elif config == 'animal-object':
        if 'with' in dir_name:
            match = re.match(r'a (\w+) with a (\w+)', dir_name)
        else:
            match = re.match(r'a (\w+) and a (\w+) (\w+)', dir_name)
    elif config == 'object-object':
        match = re.match(r'a (\w+) (\w+) and a (\w+) (\w+)', dir_name)

    if match:
        if config == 'animal-animal' or config == 'similar-subjects':
            category1 = match.group(1)
            category2 = match.group(2) 
            return category1, category2
        elif config == 'animal-object':
            category1 = match.group(1) 
            category2 = match.group(2) if 'with' in dir_name else match.group(3)
            attribute2 = "" if 'with' in dir_name else match.group(2)
            category2 = f"{attribute2}_{category2}" if attribute2 else category2
            return category1, category2
        elif config == 'object-object':
            attribute1 = match.group(1) 
            attribute2 = match.group(3) 
            category1 = match.group(2) 
            category2 = match.group(4) 
            return f"{attribute1}_{category1}", f"{attribute2}_{category2}"
    else:
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='animal-animal', type=str, help="Dataset type")
    parser.add_argument("--image_root", type=str, help="Path to the image file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--resume", default="", type=str, help="Resume from the last checkpoint")

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
        
    with open(os.path.join(output_dir, f"{out_name}{args.resume}.yaml"), 'w') as f:
        for dir_name in os.listdir(root):
            category1, category2 = decode_dir(dir_name, config=args.dataset)
            # evaluation with GPT4o and save results
            if category1 is None or category2 is None:
                print(f"Skipping {dir_name}")
                continue
            elif category1 is not None and category2 is not None:
                dir_path = os.path.join(root, dir_name)
                for image_name in os.listdir(dir_path):
                    seed = os.path.basename(image_name).split('.')[0]
                    if args.resume and f"{category1}_{category2}_{seed}" in done_keys:
                        continue
                    save_key = f"{category1}_{category2}_{seed}"
                    print(save_key)
                    image_path = os.path.join(dir_path, image_name)
                    answers = eval_gpt4o(image_path, category1, category2, config=args.dataset)
                    results = decode_answers(answers, nlp, config=args.dataset)
                    # save results
                    yaml_data = {save_key: results}
                    yaml.dump(yaml_data, f)

