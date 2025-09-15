# components.py
import copy
import random
import numpy as np
import networkx as nx
from enum import Enum
from typing import List
from tensorflow import keras  # TF2-unified Keras

class ModuleComposition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    CONV = "conv2d"
    DENSE = "dense"
    OUTPUT = "output"
    COMPILER = "compiler"

class ComponentParameters(Enum):
    CONV2D = (
        keras.layers.Conv2D,
        {"filters": ([8, 16], "int"), "kernel": ([3, 5, 7], "list"), "stride": ([1, 2, 3], "list")},
    )
    MAXPOOLING2D = (keras.layers.MaxPooling2D, {"kernel": [3, 5, 7]})
    FLATTEN = (keras.layers.Flatten, 0)
    DENSE = (keras.layers.Dense, {"units": 128, "activation": "relu"})

class Component:
    def __init__(
        self,
        representation,
        keras_component=None,
        complementary_component=None,
        keras_complementary_component=None,
        component_type=None,
    ):
        self.representation = representation
        self.keras_component = keras_component
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component
        self.component_type = component_type

    def get_layer_size(self):
        if self.component_type == "conv2d":
            return self.representation[1]["filters"]
        if self.component_type == "dense":
            return self.representation[1]["units"]
        return 0

class Module:
    def __init__(
        self,
        components: dict,
        layer_type: ModuleComposition = ModuleComposition.INPUT,
        mark=None,
        component_graph=None,
        parents=None,
    ):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.mark = mark
        self.weighted_scores = [99, 0]
        self.score_log = []
        self.species = None
        self.parents = parents
        self.use_count = 0

    def __getitem__(self, item):
        return self.components[item]

    def get_module_size(self):
        s = 0
        for node in self.component_graph.nodes():
            s += self.component_graph.nodes[node]["node_def"].get_layer_size()
        return s

    def get_kmeans_representation(self):
        node_count = len(self.component_graph.nodes())
        edge_count = len(self.component_graph.edges())
        module_size = self.get_module_size()
        return node_count, edge_count, module_size

    def update_scores(self, scores):
        self.score_log.append(scores)

    def update_weighted_scores(self):
        if len(self.score_log) > 0:
            arr = np.array(self.score_log)
            self.weighted_scores = [arr[:, 0].mean(), arr[:, 1].mean()]

    def simple_crossover(self, parent_2, mark):
        p1 = self.component_graph
        p2 = parent_2.component_graph
        child_graph = self.component_graph.copy()
        p1_nodes = list(p1.nodes())
        p2_nodes = list(p2.nodes())
        c_nodes = list(child_graph.nodes())
        for n in range(len(c_nodes)):
            parent_1_node = p1.nodes[p1_nodes[n]]["node_def"]
            parent_2_node = (
                p2.nodes[p2_nodes[n]]["node_def"] if n < len(p2.nodes()) else p2.nodes[random.choice(p2_nodes)]["node_def"]
            )
            if parent_2_node.component_type == parent_1_node.component_type:
                chosen = random.choice([parent_1_node, parent_2_node])
                child_graph.nodes[c_nodes[n]]["node_def"] = copy.deepcopy(chosen)
        return Module(None, layer_type=self.layer_type, mark=mark, component_graph=child_graph)

class Blueprint:
    def __init__(self, modules: List[Module], input_shape=None, module_graph=None, mark=None, parents=None):
        self.modules = modules
        self.input_shape = input_shape
        self.module_graph = module_graph
        self.mark = mark
        self.weighted_scores = [99, 0]
        self.score_log = []
        self.species = None
        self.parents = parents
        self.use_count = 0

    def __getitem__(self, item):
        return self.modules[item]

    def get_blueprint_size(self):
        s = 0
        for node in self.module_graph.nodes():
            s += self.module_graph.nodes[node]["node_def"].get_module_size()
        return s

    def get_kmeans_representation(self):
        node_count = len(self.module_graph.nodes())
        edge_count = len(self.module_graph.edges())
        return node_count, edge_count, self.get_blueprint_size()

    def update_scores(self, scores):
        self.score_log.append(scores)
        for node in self.module_graph.nodes():
            self.module_graph.nodes[node]["node_def"].update_scores(scores)

    def update_weighted_scores(self):
        if len(self.score_log) > 0:
            arr = np.array(self.score_log)
            self.weighted_scores = [arr[:, 0].mean(), arr[:, 1].mean()]

    def simple_crossover(self, parent_2, mark):
        p1 = self.module_graph
        p2 = parent_2.module_graph
        child_graph = self.module_graph.copy()
        p1_nodes = list(p1.nodes())
        p2_nodes = list(p2.nodes())
        c_nodes = list(child_graph.nodes())
        for n in range(len(c_nodes)):
            n1 = p1.nodes[p1_nodes[n]]["node_def"]
            n2 = p2.nodes[p2_nodes[n]]["node_def"] if n < len(p2.nodes()) else p2.nodes[random.choice(p2_nodes)]["node_def"]
            if n2.layer_type == n1.layer_type:
                chosen = random.choice([n1, n2])
                child_graph.nodes[c_nodes[n]]["node_def"] = copy.deepcopy(chosen)
        return Blueprint(None, input_shape=self.input_shape, module_graph=child_graph, mark=mark)
