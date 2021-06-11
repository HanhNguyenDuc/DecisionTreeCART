import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime
import os
from sklearn import tree, metrics



# Dự đoán được với nhãn dạng catergorical (Cần phải làm)
# Build thành cây, visualize được

def get_attribute_type(col):
    unique =  np.unique(col)
    unique_cmp = np.arange(len(unique))
    
    if np.sum((unique_cmp - unique)**2) == 0:
        return 'catetogrical'
    return 'numerical'

class DecisionTree():

    def __init__(self, criterion="entropy", max_depth=None, labels=None, save_path=None):
        self.criterion=criterion
        self.max_depth = 999999999 if max_depth is None else max_depth
        self.decision_tree=[]
        self.adjacency_list=[]
        self.save_path = "" if save_path is None else save_path

    def zip_model(self):
        return [self.labels, ]

    
    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array([[y] for y in y_train])
        self.labels = np.unique(self.y_train)


        self.dataset = np.concatenate((self.x_train, self.y_train), axis=1)
        self.attribute_num = x_train.shape[1]
        self.attributes = []


        self.attributes = []
        for i in range(self.x_train.shape[1]):
            col = self.x_train[:,i]
            self.attributes.append((np.min(col) - 1, np.max(col)+1))
        self.bound = [(attr[0], attr[1]) for attr in self.attributes]
        print("[INFO] bound: {}".format(self.bound))
        # print("[INFO] attribute_num: {}".format(self.attribute_num))
        # print("[INFO] x_train.shape = {}".format(self.x_train.shape))
        # print("[INFO] y_train.shape = {}".format(self.y_train.shape))
        # print("[INFO] dataset.shape = {}".format(self.dataset.shape))

        # Bắt đầu xây dựng cây
        self.root_nodes = self.DFS(self.dataset, self.bound, 0, -1)

        # Lưu model
        # weight_name = "weight_{}.npy".format(datetime.now().strftime("%Y%m%d%H%M%S"))
        # weight_path = os.path.join(self.save_path, weight_name)
        # np.save(weight_path, self.root_nodes)

    def load_model(self, path):
        # Load model
        self.root_nodes = np.load(path, allow_pickle=True)


    def calc_gini(self, dataset):
        # kiểm tra size của data hiện tại, nếu = 0 thì trả về giá trị entropy = 0
        if dataset.size == 0:
            return 0.0
        
        # đếm số lượng các mẫu thuộc các nhãn tương ứng
        spec_count = np.zeros(len(self.labels))
        total_row = 0
        gini = 1.0
        for index, row in enumerate(dataset):
            total_row += 1
            for enum in range(len(self.labels)):
                # print(row[-1])
                if row[-1] == self.labels[enum]:
                    spec_count[enum] += 1
        

        # tính xác xuất  ứng với mỗi nhãn trong dataset đang xét và gini
        for enum in range(len(self.labels)):
            if spec_count[enum] == 0:
                continue
            prob = spec_count[enum] / total_row
            gini -= prob ** 2
        return gini

    def calc_gini_gain(self, dataset, col_label, lbound):
        # tính gini trước khi chia nhánh
        gini_gain = 0.0

        total_row = 0
        true_count = 0
        false_count = 0

        for index, row in enumerate(dataset):
            total_row += 1
            if float(row[col_label]) <= lbound:
                true_count += 1
            else:
                false_count += 1
        
        # tính gini của các giá trị thỏa mãn điều kiện <= ngưỡng
        gini_gain += (true_count / total_row) * self.calc_gini(dataset[dataset[:,col_label] <= lbound])
        # tính gini của các giá trị không thỏa mãn điều kiện <= ngưỡng
        gini_gain += (false_count / total_row) * self.calc_gini(dataset[dataset[:,col_label] > lbound])
        return gini_gain

    def DFS(self, dataset, bound, index, depth):  # sourcery no-metrics
        # nếu toán bộ nhãn trong tập dataset đang xét đều giống nhau
        # thì coi nó là nút lá và trả về nhãn tương ứng
        for spec in self.labels:
            if len(dataset[dataset[:,-1] == spec]) == len(dataset):
                return -1, -1, spec, -1, -1

        # nếu độ sâu của cây > độ sâu cho phép, coi đây là một nút lá và trả về nhãn chiếm đa số
        if depth >= self.max_depth:
            return self.find_majority_of_data(dataset)
        
        
        mingini_gain = 10
        # nhãn, vị trí, giá trị ngưỡng
        best_criteria = ("N/A", -1, -1.0)
        colid = 0

        calc_gini = self.calc_gini(dataset)

        # xét qua từng cột thuộc tính
        for col_label in range(dataset.shape[1]):
            
            # nếu cột hiện tại là nhãn thì bỏ qua
            if col_label == dataset.shape[1] -1:
                continue


            for lbound in np.arange(bound[colid][0], bound[colid][1], 0.1):
                # tính toán gini gain, nếu lớn hơn mingini_gain hiện tại thì gán maxinfo_gain hiện tại
                # bằng chính nó
                info_gain = self.calc_gini_gain(dataset, col_label, float(lbound))
                print("col-label: {} - gini_gain: {} - lbound: {}".format(col_label, info_gain, lbound))
                if info_gain <= mingini_gain:
                    mingini_gain = info_gain
                    best_criteria = (col_label, colid, float(lbound))
            colid += 1

        # nếu gain của các nhánh > gain của node hiện tại thì coi như đây là nút gốc và trả về nhãn chiếm đa số
        if calc_gini - mingini_gain <= 0:
            return self.find_majority_of_data(dataset)

        # print("maxinfo_gain: {}".format(maxinfo_gain))
        # print("Bound change: {} - {} - {}, lbound = {}".format(best_criteria[0], bound[best_criteria[1]][0], bound[best_criteria[1]][1], best_criteria[2]))
        

        child_nodes = []


        # gọi hàm đệ quy xét nhánh 1 (nhánh có thuộc tính hiện tại <= ngưỡng - tức là nhánh thỏa mãn điều kiện hiện tại)
        subset1 = dataset[dataset[:,best_criteria[0]]<=best_criteria[2]]
        newbound = bound.copy()
        newbound[best_criteria[1]] = (newbound[best_criteria[1]][0], best_criteria[2] - 0.1)
        left_node = self.DFS(subset1, newbound, index, depth+1)
        if left_node:
            child_nodes.append(left_node)

        # gọi hàm đệ quy xét nhánh 2 (nhánh có thuộc tính hiện tại > ngưỡng - tức là nhánh không thỏa mãn điều kiện hiện tại)
        subset2 = dataset[dataset[:,best_criteria[0]]>best_criteria[2]]
        newbound = bound.copy()
        newbound[best_criteria[1]] = (best_criteria[2], newbound[best_criteria[1]][1])
        right_node = self.DFS(subset2, newbound, index, depth+1)
        if right_node:
            child_nodes.append(right_node)

        return (best_criteria[1], best_criteria[2], child_nodes, mingini_gain, calc_gini)

    def find_majority_of_data(self, dataset):
        labels = np.unique(dataset[:, -1])
        counts = [len(dataset[dataset[:, -1] == label]) for label in labels]
        max_label_count = -1
        max_label = -1
        label_count_dict = dict(zip(labels, counts))
        for label_count, value_ in label_count_dict.items():
            if value_ > max_label_count:
                max_label_count = label_count_dict[label_count]
                max_label = label_count
        return -1, -1, max_label, -1, -1

    def go_in_node(self, x, node):
        col = node[0]
        bound = node[1]
        # print("Node: {}".format(node[2]))
        # print("Child1 node: {} - {}".format(node[2][0][0], node[2][0]))
        # print("Child1 node: {} - {}".format(node[2][1][0], node[2][1]))
        if x[col] < bound:
            if node[2][0][0] == -1:
                return node[2][0][2]
            else:
                return self.go_in_node(x, node[2][0])
        elif node[2][1][0] == -1:
            return node[2][1][2]
        else:
            return self.go_in_node(x, node[2][1])

    def predict(self, x):
        return self.go_in_node(x, self.root_nodes)

    def display_node(self, node, head, depth):
        if node[0] == -1:
            for _ in range(depth):
                print("|  ", end="")
            print("---label: {}".format(node[2]))
            return
        
        for _ in range(depth):
            print(head, end="")
        print("---col: {} <= ".format(node[0]) + "{0:.2f}".format(node[1]) + ", min-gini-gain: {0:.1f}".format(node[3]), ", node-gain: {0:.1f}".format(node[4]))
        self.display_node(node[2][0], head, depth+1)

        for _ in range(depth):
            print(head, end="")
        print("---col: {} > ".format(node[0]) + "{0:.2f}".format(node[1]) + ", min-gini-gain: {0:.1f}".format(node[3]), ", node-gain: {0:.1f}".format(node[4]))
        self.display_node(node[2][1], head, depth+1)

    def display_tree(self):
        print("[INFO] Start of display tree ----------------------")
        self.display_node(self.root_nodes, "|  ", 0)

        print("[INFO] End of display tree ----------------------")

    def evaluate(self, x_test, y_test):
        true_num = 0
        for i, x in enumerate(x_test):
            y_predict = self.predict(x)
            if y_predict == y_test[i]:
                true_num+=1
        print("[INFO] accuracy: {0:.3f}".format(true_num/len(x_test)))


if __name__ == "__main__":
    # load data
    start_time = time.time()

    data = pd.read_csv(
        "iris.csv",
    )
    data = data.sample(frac=1)
    total_data = len(data)
    x_train = np.array((data.iloc[:-50,1:-1].values), dtype='float32')
    x_test = np.array((data.iloc[-50:,1:-1].values), dtype='float32')
    y_train = np.unique(data.iloc[:-50,-1].values, return_inverse=True)[1]
    y_test = np.unique(data.iloc[-50:,-1].values, return_inverse=True)[1]


    # bound = [(0.0, 10.0) for i in range(4)]

    dt = DecisionTree()
    dt.fit(x_train, y_train)
    dt.display_tree()
    dt.evaluate(x_test, y_test)
    clf = tree.DecisionTreeClassifier(criterion="gini")
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    text_representation = tree.export_text(clf)
    print(text_representation)
    # Test entropy function

    # dt.DFS(data, bound, 0, -1)
