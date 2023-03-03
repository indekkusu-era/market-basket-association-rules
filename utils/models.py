import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from dataclasses import dataclass

@dataclass(frozen=True)
class AssociationRule:
    if_: tuple
    then_: tuple
    support: float
    confidence: float
    lift: float

class AssociaitionRulesModel:
    def __init__(self, support_lower: float, confidence_lower: float, lift_lower: float, n_products=3):
        self.support_lower = support_lower
        self.confidence_lower = confidence_lower
        self.lift_lower = lift_lower
        self.n_products = n_products
    
    def generate_permutations(self, columns: list, n: int):
        if n == 1:
            return [(k,) for k in columns]
        permutations = []
        for i, c in enumerate(columns):
            permutations.extend([(c,) + p for p in self.generate_permutations(columns[i+1:], n-1)])
        return permutations

    def generate_all_possible_combinations(self, columns: list, n: int):
        combinations = []
        for i in range(1, n+1):
            combinations.extend(self.generate_permutations(columns, i))
        return combinations 
    
    @staticmethod
    def calculate_combination_probability(df: pd.DataFrame, combination: tuple):
        logic = None
        for product in combination:
            if logic is None:
                logic = df[product] == 1
                continue
            logic &= (df[product] == 1)
        return len(df[logic]) / len(df)
    
    def calculate_all_probabilities(self, X: pd.DataFrame):
        self.probabilities_ = {}
        for combination in self.all_possible_combinations_:
            self.probabilities_[frozenset(combination)] = self.calculate_combination_probability(X, combination)
        return self.probabilities_
    
    def generate_all_rules(self):
        rules = []
        rules_generator = list(filter(lambda x: len(x) > 1, self.all_possible_combinations_))
        for combination in rules_generator:
            rules.extend(self.generate_combination_rules(combination))
        self.all_rules_ = list(set(rules))
        return self.all_rules_
    
    def generate_combination_rules(self, combination: tuple):
        possible_comb = self.generate_all_possible_combinations(combination, self.n_products)
        rules = []
        for if_ in possible_comb:
            for then_ in possible_comb:
                if set(if_).intersection(set(then_)).__len__() != 0:
                    continue
                rules.append((if_, then_))
        return rules
    
    def calculate_rule_metric(self, rule: tuple):
        if_, then_ = rule
        support = self.probabilities_[frozenset(if_ + then_)]
        confidence = 0
        lift = 0
        if self.probabilities_[frozenset(if_)] != 0:
            confidence = self.probabilities_[frozenset(if_ + then_)] / self.probabilities_[frozenset(if_)]
            if self.probabilities_[frozenset(then_)] != 0:
                lift = self.probabilities_[frozenset(if_ + then_)] / (self.probabilities_[frozenset(if_)] * self.probabilities_[frozenset(then_)])
        association_rule = AssociationRule(if_=if_, then_=then_, support=support, confidence=confidence, lift=lift)
        return association_rule
    
    def calculate_metrics(self):
        self.association_metrics_ = []
        for rule in self.all_rules_:
            self.association_metrics_.append(self.calculate_rule_metric(rule))
        return self.association_metrics_
    
    def association_rules(self):
        return list(filter(lambda x: x.support > self.support_lower and x.confidence > self.confidence_lower and x.lift > self.lift_lower, self.association_metrics_))

    def fit(self, X: pd.DataFrame):
        self.feature_names_ = X.columns
        self.all_possible_combinations_ = self.generate_all_possible_combinations(X.columns, self.n_products)
        self.calculate_all_probabilities(X)
        self.generate_all_rules()
        self.calculate_metrics()
        self.association_rules_ = self.association_rules()
    
    def accuracy(self, Xtest: pd.DataFrame):
        test_rules = []
        for rule in self.association_rules_:
            if_ = rule.if_; then_ = rule.then_
            support = self.calculate_combination_probability(Xtest, if_ + then_)
            if_prob = self.calculate_combination_probability(Xtest, if_)
            then_prob = self.calculate_combination_probability(Xtest, then_)
            confidence = support / if_prob if if_prob != 0 else 0
            lift = confidence / then_prob if then_prob != 0 else 0
            test_rule = AssociationRule(if_, then_, support, confidence, lift)
            test_rules.append(test_rule)
        return test_rules

class AssociationTree(AssociaitionRulesModel):
    def __init__(self, support_lower: float, confidence_lower: float, lift_lower: float, n_products=3):
        super().__init__(support_lower, confidence_lower, lift_lower, n_products)
        self.probabilities_ = {}
    
    @staticmethod
    def get_association_rules_from_tree(tree: DecisionTreeClassifier, feature_names: list, out: tuple):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]

        # get ids of child nodes
        idx = np.argwhere(left == -1)[:,0]     

        def recurse(left, right, child, lineage=None):
            if lineage is None:
                buy = np.argmax(tree.tree_.value[child]) 
                lineage = [buy]
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'

            lineage.append((parent, split, threshold[parent], features[parent]))

            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
        
        rules = []

        currule = tuple()
        for child in idx:
            for node in recurse(left, right, child):
                if isinstance(node, np.int64):
                    if node == 1:
                        rules.append((currule, out))
                    currule = tuple()
                    continue
                if node[1] == 'r':
                    currule = (node[3],) + currule
        return rules
    
    def generate_feature_name_rules(self, X, feature_name):
        tree = DecisionTreeClassifier(max_depth=self.n_products)
        y = X[feature_name]
        Xtree = X.drop(feature_name, axis=1)
        tree.fit(Xtree, y)
        return self.get_association_rules_from_tree(tree, tree.feature_names_in_, out=(feature_name,))

    def generate_all_rules(self, X: pd.DataFrame):
        self.all_rules_ = []
        for feature_name in self.feature_names_:
            rules = self.generate_feature_name_rules(X, feature_name)
            self.all_rules_.extend(rules)
    
    def calculate_rule_metric(self, rule: tuple):
        if_, then_ = rule
        if frozenset(if_ + then_) not in self.probabilities_.keys():
            p = self.calculate_combination_probability(self._X, if_ + then_)
            self.probabilities_[frozenset(if_ + then_)] = p
        if frozenset(if_) not in self.probabilities_.keys():
            p = self.calculate_combination_probability(self._X, if_)
            self.probabilities_[frozenset(if_)] = p
        if frozenset(then_) not in self.probabilities_.keys():
            p = self.calculate_combination_probability(self._X, then_)
            self.probabilities_[frozenset(then_)] = p
        support = self.probabilities_[frozenset(if_ + then_)]
        confidence = 0
        lift = 0
        if self.probabilities_[frozenset(if_)] != 0:
            confidence = self.probabilities_[frozenset(if_ + then_)] / self.probabilities_[frozenset(if_)]
            if self.probabilities_[frozenset(then_)] != 0:
                lift = self.probabilities_[frozenset(if_ + then_)] / (self.probabilities_[frozenset(if_)] * self.probabilities_[frozenset(then_)])
        association_rule = AssociationRule(if_=if_, then_=then_, support=support, confidence=confidence, lift=lift)
        return association_rule
    
    def fit(self, X: pd.DataFrame):
        self._X = X.copy()
        self.feature_names_ = X.columns
        self.generate_all_rules(X)
        self.calculate_metrics()
        self.association_rules_ = self.association_rules()
