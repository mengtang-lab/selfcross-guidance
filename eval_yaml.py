import yaml
import argparse
import os


def calculate_true_frequencies(yaml_filename, num_questions=2):
    # Load the data from the YAML file
    print(f"Analyzing {yaml_filename}...")
    with open(yaml_filename, 'r') as file:
        data = yaml.safe_load(file)

    # Initialize a dictionary to count the true values for each question
    question_counts = {'diversity': 0, 'recognizable': 0, 'overall': 0}
    total_entries = 0  # Count the total number of entries for normalization

    reeval_keys = []
    for key, questions in data.items():
        if len(questions.keys()) != num_questions:
            reeval_keys.append(key)
            print(f"Error: {key} does not have correct number of questions")
            continue
        if isinstance(questions, dict):  # Check if the value is a dictionary
            total_entries += 1
            # for question, value in questions.items():
            #     if question not in question_counts.keys():
            #         question_counts[question] = 0
            #     # not 'content_mixture' is better and what we want
            #     if question == 'diversity' and value is False:
            #         question_counts[question] += 1
            #     elif question != 'diversity' and value is True:
            #         question_counts[question] += 1

            if questions['diversity']:
                question_counts['diversity'] += 1

            if questions['recognizable']:
                question_counts['recognizable'] += 1

            if questions['recognizable'] and questions['diversity']:
                question_counts['overall'] += 1

    # Print the frequency of `true` for each question
    print("Frequency of `true` values for each question:")
    for question, count in question_counts.items():
        frequency = count / total_entries
        print(f"{question}: {count} true ({frequency:.2%} frequency)")

    # Print overall statistics
    # overall_frequency = sum(question_counts.values()) / (total_entries * len(question_counts.keys()))
    # print(f"\nOverall frequency of `true` values: {overall_frequency:.2%}\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--yaml_filename', default='./SD3/outputsFalse.yaml', type=str)
    args = args.parse_args()
    if not os.path.exists(args.yaml_filename): print(f"Error: {args.yaml_filename} does not exist")
    calculate_true_frequencies(yaml_filename=args.yaml_filename)


# Analyzing ./SD3_1119/outputs.yaml...
# Frequency of `true` values for each question:
# diversity: 180 true (90.00% frequency)
# recognizable: 167 true (83.50% frequency)
# overall: 167 true (83.50% frequency)

# Analyzing ./SD3/outputs.yaml...
# Frequency of `true` values for each question:
# diversity: 123 true (61.50% frequency)
# recognizable: 187 true (93.50% frequency)
# overall: 115 true (57.50% frequency)

# prompt eng
# Analyzing ./SD3_prompt_eng/outputs.yaml...
# Frequency of `true` values for each question:
# diversity: 160 true (80.00% frequency)
# recognizable: 189 true (94.50% frequency)
# overall: 151 true (75.50% frequency)