import numpy as np
import pandas as pd
import math


def check_purity(data):
    class_column = data[:, -1]
    count_of_0 = 0

    for e in class_column:
        if e == 0:
            count_of_0 = count_of_0 + 1

    if (len(class_column) == count_of_0) | (count_of_0 == 0):
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    count_of_0 = 0

    for i in label_column:
        if i == 0:
            count_of_0 = count_of_0 + 1

    if count_of_0 > len(label_column) - count_of_0:
        return 0
    else:
        return 1


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = [0, 1]

    return potential_splits


def split_data(data, split_column):
    split_column_values = data[:, split_column]

    data_left = data[split_column_values == 0]
    data_right = data[split_column_values == 1]

    return data_left, data_right


def calculate_entropy(data):
    label_column = data[:, -1]
    count_of_0 = count_of_1 = 0

    for i in label_column:
        if i == 0:
            count_of_0 = count_of_0 + 1
        if i == 1:
            count_of_1 = count_of_1 + 1

    total_count = count_of_0 + count_of_1

    if total_count == 0:
        return 0

    p1 = (count_of_0 * 1.0) / total_count
    p2 = (count_of_1 * 1.0) / total_count
    entropy = 0

    if p1 > 0:
        entropy = p1 * math.log(p1, 2)
    if p2 > 0:
        entropy = entropy + p2 * math.log(p2, 2)

    return entropy * -1


def calculate_overall_entropy(data_left, data_right):
    n = len(data_left) + len(data_right)

    if n == 0:
        return 0

    p_data_left = len(data_left) / n
    p_data_right = len(data_right) / n

    overall_entropy = (p_data_left * calculate_entropy(data_left) + p_data_right * calculate_entropy(data_right))

    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    best_split_column = 0
    for column_index in potential_splits:
        data_left, data_right = split_data(data, split_column=column_index)
        current_overall_entropy = calculate_overall_entropy(data_left, data_right)

        if current_overall_entropy <= overall_entropy:
            overall_entropy = current_overall_entropy
            best_split_column = column_index

    return best_split_column


def decision_tree_algorithm(df, counter=0, min_samples=0, max_depth=10):
    
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification

    else:
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column = determine_best_split(data, potential_splits)
        data_left, data_right = split_data(data, split_column)
        
        feature_name = COLUMN_HEADERS[split_column]
        question = "{}".format(feature_name)
        sub_tree = {question: []}

        no_answer = decision_tree_algorithm(data_left, counter, min_samples, max_depth)
        yes_answer = decision_tree_algorithm(data_right, counter, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(no_answer)
            sub_tree[question].append(yes_answer)

        return sub_tree


df = pd.read_csv("/Users/nimish/Documents/ML/temp_dataset/training_set.csv")
tree = decision_tree_algorithm(df)
print(tree)
