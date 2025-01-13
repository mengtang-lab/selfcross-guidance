import argparse
import yaml
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def calculate_true_frequencies(yaml_filename, num_questions=5):
    # Load the data from the YAML file
    print(f"Analyzing {yaml_filename}...")
    with open(yaml_filename, 'r') as file:
        data = yaml.safe_load(file)
    
    # Initialize a dictionary to count the true values for each question
    question_counts = {'both_appear': 0, 'both_recognizable': 0}
    total_entries = 0  # Count the total number of entries for normalization
    
    reeval_keys = []
    for key, questions in data.items():
        if len(questions.keys()) != num_questions:
            reeval_keys.append(key)
            print(f"Error: {key} does not have correct number of questions")
            continue
        if isinstance(questions, dict):  # Check if the value is a dictionary
            total_entries += 1
            for question, value in questions.items():
                if question not in question_counts.keys():
                    question_counts[question] = 0
                # not 'content_mixture' is better and what we want
                if question == 'content_mixture' and value is False:
                    question_counts[question] += 1
                elif question != 'content_mixture' and value is True: 
                    question_counts[question] += 1
            if questions['A_appear'] and questions['B_appear']:
                question_counts['both_appear'] += 1
            if questions['A_recognizable'] and questions['B_recognizable']:
                question_counts['both_recognizable'] += 1

    # Print the frequency of `true` for each question
    print("Frequency of `true` values for each question:")
    for question, count in question_counts.items():
        frequency = count / total_entries
        print(f"{question}: {count} true ({frequency:.2%} frequency)")
    
    # Print overall statistics
    overall_frequency = sum(question_counts.values()) / (total_entries * len(question_counts.keys()))
    print(f"\nOverall frequency of `true` values: {overall_frequency:.2%}\n")

#   # remove reeval keys from data and rewrite the yaml file
#   if reeval_keys:
#       for key in reeval_keys:
#           data.pop(key)
#       with open(yaml_filename, 'w') as file:
#           yaml.dump(data, file)
#           print(f"Removed {len(reeval_keys)} keys from {yaml_filename}\n")


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('yaml_filename', type=str)
    args = args.parse_args()
    calculate_true_frequencies(args.yaml_filename)
