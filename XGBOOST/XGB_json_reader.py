''' Use this script to deploy your XGB model in pure python. (so you can remove all XGB dependencies while deploying it on another OS)

Requirements :  - dump your trained model in .json (with a maping of your features, all feature names must not contain spaces)                         
               - in the traning phase for a classification problem, the hyperparameteres   "base_score" i.e the predictions intialization must be the default value  (0.5) 
               - input data must be passed as a list of dict when using the predict function'''
                

import math
class BoosterReader:
    def __init__(self, model_data, pred_type, base_score=0.5):
    
        if (pred_type == 'classification') & (base_score != 0.5):
            raise ValueError('For classification, set base_score as 0.5')

        self.pred_type = pred_type
        self.base_score = base_score
        self.model_data = model_data
        self.build_boosters(model_data)

    # representing all decision trees as Boosters object givin the .json file input
    def build_boosters(self, model_data):
        self.booster = []

        for booster_data in model_data:
            self.trees.append(Booster(booster_data))

    # predict over all data, data must be passed as list of dictionnaries
    def predict(self, data):
        if len(data) == 1:
            return self._predict_row(data[0])

        return [self._predict_row(row) for row in data]

    # get raw leaf prediction values for each booster, return a list for each booster
    def get_leaf_values(self, data, node_value=True):
        leaf_values = [booster.get_leaf_value(data, node_value) for booster in self.booster]
        return leaf_values

    # predict for a single observation input (summing up predicted values for each booster : they are already scaled by learning rate during training process!)
    # convert from log(odds) to probabilites for classification
    def _predict_row(self, data):
        total_leaf_value = sum([booster.get_leaf_value(data) for booster in self.booster])

        if self.pred_type == 'regression':
            prediction = total_leaf_value + self.base_score
        #return probabilities using sigmoid function as activation function
        if self.pred_type == 'classification':
            prediction = 1. / (1. + math.exp(-total_leaf_value))
        return prediction

# representation of a single booster : will store info of decision tree, output leaf values of the booster given input data 
class Booster:
    def __init__(self, node_data):
        self.build_booster(node_data)

    #build the hierarchy of the booster's nodes used to find leaf value prediction given input of a dictionary of node data
    def build_booster(self, node_data):
        self.root = Node(node_data)

    #get the leave value for an input data in the booster
    def get_leaf_value(self, data, node_value=True):
        leaf_value = self.root.leaf_value(data, node_value)
        return leaf_value

# representation of single node in a given tree : if it's a leaf, store final value, if it's a split node, store information on child nodes in children dict
class Node:
    def __init__(self, node_data):
        self.leaf = node_data['leaf'] if 'leaf' in node_data else None
        self.node_id = node_data['nodeid']
        if self.leaf is None:
            self.depth = node_data['depth']
            self.split_feature = node_data['split']
            self.threshold = node_data['split_condition'] if 'split_condition' in node_data else None
            self.yes_id = node_data['yes']
            self.no_id = node_data['no']
            self.missing_id = node_data['missing'] if 'missing' in node_data else None
            self.children = {}
            self.get_children(node_data['children'])

    #create children as dictionnary of its child nodes and their own children (hiearchy of data)
    def get_children(self, children):
        for child in children:
            self.get_child(child)
            

    #for an input of child node_data represent the child as a dict of(node_id, split_condition,  identifier, node's children info ect..) 
    def get_child(self, child):
        self.children[child['nodeid']] = Node(child)

    #get value of a node : if it's a value, return the output, if not, compare split_feature vs node's split conditions in input data, and try to find down a child node id where decision threshold condition is met 
    def leaf_value(self, data, node_value):
        if self.leaf is not None:
            if node_value:
                return self.leaf
            else:
                return self.node_id
        if self.split_feature not in data:
            return self.children[self.missing_id].leaf_value(data, node_value)
        feature_value = data[self.split_feature]
        if self.missing_id is not None:
            if math.isnan(feature_value):
                return self.children[self.missing_id].leaf_value(data, node_value)
        if self.threshold is not None:
            if feature_value >= self.threshold:
                return self.children[self.no_id].leaf_value(data, node_value)
            else:
                return self.children[self.yes_id].leaf_value(data, node_value)
        else:
            if feature_value == 0.0:
                return self.children[self.no_id].leaf_value(data, node_value)
            else:
                return self.children[self.yes_id].leaf_value(data, node_value)
