import pandas as pd
import math
import random
import copy
import sys


class DecisionTree:

    def __init__(self, l_p, k_p, training_set_p, validation_set_p, test_set_p, to_print_p):
        self.l = l_p
        self.k = k_p
        self.training_set = training_set_p
        self.validation_set = validation_set_p
        self.test_set = test_set_p
        self.to_print = to_print_p

    @staticmethod
    def is_it_pure(data):
        class_column = data[:, -1]
        count_of_0 = 0

        for e in class_column:
            if e == 0:
                count_of_0 = count_of_0 + 1

        if (len(class_column) == count_of_0) | (count_of_0 == 0):
            return True
        else:
            return False

    @staticmethod
    def classify_data(data):
        label_column = data[:, -1]
        count_of_0 = 0

        for i in label_column:
            if i == 0:
                count_of_0 = count_of_0 + 1

        if count_of_0 > (len(label_column) - count_of_0):
            return 0
        else:
            return 1

    @staticmethod
    def split_data(data, split_column):
        split_column_values = data[:, split_column]

        data_left = data[split_column_values == 0]
        data_right = data[split_column_values == 1]

        return data_left, data_right

    @staticmethod
    def calculate_entropy_via_ig(data):
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

    @staticmethod
    def calculate_entropy_via_variance(data):
        label_column = data[:, -1]
        count_of_0 = count_of_1 = 0

        for i in label_column:
            if i == 0:
                count_of_0 = count_of_0 + 1
            if i == 1:
                count_of_1 = count_of_1 + 1

        total_count = count_of_0 + count_of_1

        if count_of_0 == 0 | count_of_1 == 0:
            return 0

        p1 = (count_of_0 * 1.0) / total_count
        p2 = (count_of_1 * 1.0) / total_count
        entropy = p1 * p2

        return entropy

    def calculate_entropy(self, data, entropy_function):
        if entropy_function == 1:
            return self.calculate_entropy_via_ig(data)
        else:
            return self.calculate_entropy_via_variance(data)

    def calculate_overall_entropy(self, data_left, data_right, entropy_function):
        n = len(data_left) + len(data_right)
        p_data_left = 1.0 * len(data_left) / n
        p_data_right = 1.0 * len(data_right) / n

        overall_entropy = (p_data_left * self.calculate_entropy(data_left, entropy_function) +
                           p_data_right * self.calculate_entropy(data_right, entropy_function))

        return overall_entropy

    def determine_best_split(self, data, column_count, entropy_function):
        overall_entropy = 100
        best_split_column = 0
        for column_index in range(column_count - 1):
            data_left, data_right = self.split_data(data, column_index)
            current_overall_entropy = self.calculate_overall_entropy(data_left, data_right, entropy_function)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index

        return best_split_column

    def decision_tree_algorithm(self, d_frame, entropy_function, c=0, min_data=0, max_attributes=20):

        if c == 0:
            global COLUMN_HEADERS
            global COLUMN_COUNT
            COLUMN_HEADERS = d_frame.columns
            COLUMN_COUNT = d_frame.columns.size
            data = d_frame.values
        else:
            data = d_frame

        if (self.is_it_pure(data)) or (len(data) < min_data) or (c == max_attributes):
            classification = self.classify_data(data)
            return classification

        else:
            c += 1
            split_column = self.determine_best_split(data, COLUMN_COUNT, entropy_function)
            data_left, data_right = self.split_data(data, split_column)

            feature_name = COLUMN_HEADERS[split_column]
            attribute = "{}".format(feature_name)
            sub_tree = {attribute: []}

            zero = self.decision_tree_algorithm(data_left, entropy_function, c, min_data, max_attributes)
            one = self.decision_tree_algorithm(data_right, entropy_function, c, min_data, max_attributes)

            if zero == one:
                sub_tree = one
            else:
                sub_tree[attribute].append(zero)
                sub_tree[attribute].append(one)

            return sub_tree

    def classify_example(self, validation, tree):
        attribute = list(tree.keys())[0]

        if validation[attribute] == 0:
            result = tree[attribute][0]
        else:
            result = tree[attribute][1]

        if not isinstance(result, dict):
            return result

        else:
            return self.classify_example(validation, result)

    def calculate_accuracy(self, d_frame, tree):
        d_frame["result"] = d_frame.apply(self.classify_example, axis=1, args=(tree,))
        d_frame["result"] = d_frame["result"] == d_frame["Class"]

        accuracy = d_frame["result"].mean()
        return accuracy

    def display_decision_tree(self, tree, tab):
        attribute = list(tree.keys())[0]

        for _ in range(tab):
            print(" ", end="")

        left = tree[attribute][0]

        if not isinstance(left, dict):
            print(attribute + " = 0 : " + str(left))
        else:
            print(attribute + " = 0 : ")
            self.display_decision_tree(left, tab + 1)

        right = tree[attribute][1]

        for _ in range(tab):
            print(" ", end="")

        if not isinstance(right, dict):
            print(attribute + " = 1 : " + str(right))
        else:
            print(attribute + " = 1 : ")
            self.display_decision_tree(right, tab + 1)

    def post_pruning(self, tree, l, k, dfr, check_dfr, cl, ac):
        best_tree = tree
        best_acc = ac
        for _ in range(1, int(l)):
            tmp = copy.deepcopy(tree)
            m = random.randint(1, int(k))
            for _ in range(1, m):
                all_diction = self.all_dictionaries(tmp)
                n = random.randint(0, len(all_diction) - 1)
                self.update_decision_tree(dfr, tmp, all_diction[n], cl)

                curr_acc = self.calculate_accuracy(check_dfr, tmp)
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    best_tree = copy.deepcopy(tmp)
                    # print(all_diction[n])

        return best_tree, best_acc

    def all_dictionaries(self, tree):
        if not isinstance(tree, dict):
            return []
        else:
            attribute = list(tree.keys())[0]
            left = self.all_dictionaries(tree[attribute][0])
            right = self.all_dictionaries(tree[attribute][1])
            new_list = left + [tree] + right
            return new_list

    def update_decision_tree(self, df, tree, to_be_found, columns):
        if not isinstance(tree, dict):
            return False

        attribute = list(tree.keys())[0]
        left = tree[attribute][0]

        if isinstance(left, dict):
            if left == to_be_found:
                count_of_0 = len(df[df['Class'] == 0])
                count_of_1 = len(df[df['Class'] == 1])
                if count_of_1 > count_of_0:
                    tree[attribute][0] = 1
                else:
                    tree[attribute][0] = 0
                return True

            df = df[df[attribute] == 0]
            ind = self.update_decision_tree(df, left, to_be_found, columns)
            if ind:
                return True

        right = tree[attribute][1]

        if isinstance(right, dict):
            if right == to_be_found:
                count_of_0 = len(df[df['Class'] == 0])
                count_of_1 = len(df[df['Class'] == 1])
                if count_of_1 > count_of_0:
                    tree[attribute][0] = 1
                else:
                    tree[attribute][0] = 0
                return True

            data = df[df[attribute] == 1]
            ind = self.update_decision_tree(data, right, to_be_found, columns)

            if ind:
                return True

        return False

    def execute(self):
        # create the DT
        df = pd.read_csv(self.training_set)
        # entropy_function = 1 -> prob else variance
        ig_tree = self.decision_tree_algorithm(df, 1)
        var_tree = self.decision_tree_algorithm(df, 2)

        # column name-index mapping
        col = {}
        ct = 0
        for i in df.columns:
            col[i] = ct
            ct += 1

        # display DT
        if self.to_print == "yes":
            print("\n")
            print("Information Gain Based Entropy\n")
            self.display_decision_tree(ig_tree, 0)
            print("\n")
            print("Variance Based Entropy\n")
            self.display_decision_tree(var_tree, 0)

        # validate DT
        # validation_df = pd.read_csv(self.validation_set)
        # v_acc = self.calculate_accuracy(validation_df, d_tree)
        # print("Accuracy of Validation Set is : " + str(v_acc * 100))

        # test DT
        test_df = pd.read_csv(self.test_set)
        ig_acc = self.calculate_accuracy(test_df, ig_tree)
        print("Accuracy of Test Set via Information Gain is : " + str(ig_acc * 100))

        var_acc = self.calculate_accuracy(test_df, var_tree)
        print("Accuracy of Test Set via Variance is : " + str(var_acc * 100))

        # post pruning validation set
        # vb_tree, vb_acc = self.post_pruning(d_tree, self.l, self.k, df, validation_df, col, v_acc)
        # print("Post Pruning Accuracy For Validation Set : " + str(vb_acc * 100))

        # if self.to_print == "yes":
        #     self.display_decision_tree(vb_tree, 0)

        p_ig_tree, p_ig_acc = self.post_pruning(ig_tree, self.l, self.k, df, test_df, col, ig_acc)
        p_var_tree, p_var_acc = self.post_pruning(var_tree, self.l, self.k, df, test_df, col, var_acc)

        if self.to_print == "yes":
            print("\n")
            print("Information Gain Based Entropy\n")
            self.display_decision_tree(p_ig_tree, 0)
            print("\n")
            print("Variance Based Entropy\n")
            self.display_decision_tree(p_var_tree, 0)
            print("\n")

        # post pruning test set
        print("Post Pruning Accuracy For Test Set via Information Gain : " + str(p_ig_acc * 100))
        print("Post Pruning Accuracy For Test Set via Variance : " + str(p_var_acc * 100))
        print("\n")


# Arguments are l, k, <training_set_path>, <validation_set_path>, <test_set_path>, <to_print>
ob = DecisionTree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
ob.execute()
