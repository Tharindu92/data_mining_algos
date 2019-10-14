import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''

        self.root = self.split_node(X,y,1)
        pass

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''
        predicts = []
        for x in X:
            predicts.append(self.predict_record(x))
        return predicts

    def predict_record(self, X):
        tree = self.root
        while not isinstance(tree, float):
            if X[tree['splitting_variable']] > tree['splitting_threshold']:
                tree = tree['right']
            else:
                tree = tree['left']

        return tree

    def sort_columns(self, X, y, j):
        col = []
        for i in range(len(X)):
            item = []
            item.append(X[i][j])
            item.append(y[i])
            col.append(item)
        col.sort()

        return col

    def split_node(self, X, y ,I):
        if (len(y) < self.min_samples_split) or (I >= self.max_depth):
            return sum(y)/len(y)

        minGini = 100000000000
        J = -1
        for i in range(len(X[0])):
            X2 = X[:]
            y2 = y[:]
            C = self.sort_columns(X2,y2,i)
            Cy = []
            Cx = []
            for c in C:
                Cy.append(c[1])
                Cx.append(c[0])
            gini, s = self.mini_gini(Cy, Cx)
            if gini < minGini:
                minGini = gini
                J = i
                S = s

        rightX, rightY, leftX, leftY = self.split_to_left_right(X,y,J, S)
        tree = {}
        tree['splitting_variable'] = J
        tree['splitting_threshold'] = S
        if (len(leftY) > 0):
            # print(I)
            tree['left'] = self.split_node(leftX,leftY, I +1)
        if (len(rightY) > 0):
            tree['right'] = self.split_node(rightX,rightY, I+1)

        return tree

    def split_to_left_right(self, X, y, J, S):
        rightX = []
        rightY = []
        leftX = []
        leftY = []
        for i in range(len(y)):
            if S < X[i][J]:
                rightX.append(X[i])
                rightY.append(y[i])
            else:
                leftX.append(X[i])
                leftY.append(y[i])
        return rightX, rightY, leftX, leftY

    def mini_gini(self, y, x):
        minGini = 100000000
        minLx = []
        for j in range(len(y) - 1):
            right = y[j + 1:]
            left = y[:j + 1]
            leftX = x[:j + 1]
            gini = self.gini(right, left)
            if gini < minGini:
                minGini = gini
                minLx = leftX
        return minGini, minLx[len(minLx) - 1]

    def gini(self, right, left):
        meanR = sum(right) / len(right)
        meanL = sum(left) / len(left)

        gini = sum((xi - meanR) ** 2 for xi in right)
        gini = gini + sum((xi - meanL) ** 2 for xi in left)

        return gini

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0


def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) / np.maximum(1e-8, abs(sample_output))).mean()
    if rel_error <= 1e-5:
        return 1
    else:
        return 0

# For test
if __name__=='__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")

            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")



