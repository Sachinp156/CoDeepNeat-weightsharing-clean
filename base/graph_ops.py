# graph_ops.py
import logging
import random

import networkx as nx
import matplotlib.pyplot as plt

# shared base path for images
try:
    from .config import basepath
except Exception:
    from kerascodeepneat import basepath  # type: ignore

# core structures
try:
    from .structures import (
        Component,
        Module,
        Blueprint,
        ModuleComposition,
    )
except Exception:
    # fallback to your current single-file namespace
    from kerascodeepneat import Component, Module, Blueprint, ModuleComposition  # type: ignore


class GraphOperator:
    """
    Responsible for sampling random components/modules/blueprints and mutating graphs.
    """

    count = 0  # just for unique plot filenames

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_parameter_def(possible_parameters, parameter_name):
        """
        Sample a parameter value for a given parameter_name from 'possible_parameters'.
        possible_parameters[parameter_name] = (values, type) where type in {'int','float','list'}
        """
        values, ptype = possible_parameters[parameter_name]
        if ptype == "int":
            return random.randint(values[0], values[1])
        if ptype == "float":
            return (values[1] - values[0]) * random.random() + values[0]
        if ptype == "list":
            return random.choice(values)
        raise ValueError(f"Unknown parameter type '{ptype}' for '{parameter_name}'")

    @staticmethod
    def _safe_layout_draw(graph):
        """
        Draw with graphviz if available; otherwise fall back to spring_layout.
        """
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
        except Exception:
            pos = nx.spring_layout(graph, seed=7)
        nx.draw(graph, pos, with_labels=True, font_weight="bold", font_size=6)

    @classmethod
    def save_graph_plot(cls, filename, graph):
        """
        Save a small visual of the graph to images/. Robust to missing pygraphviz.
        """
        plt.figure()
        cls._safe_layout_draw(graph)
        plt.tight_layout()
        try:
            plt.savefig(f"{basepath}/images/{filename}", format="PNG", bbox_inches="tight")
        except Exception:
            # last-ditch: try current directory
            plt.savefig(filename, format="PNG", bbox_inches="tight")
        finally:
            plt.close()

    # ------------------------------------------------------------------ #
    # random generators
    # ------------------------------------------------------------------ #
    def random_component(self, possible_components, possible_complementary_components=None):
        """
        Build a Component object by sampling a layer type + its parameters.
        """
        component_type = random.choice(list(possible_components))
        component_def, possible_params = possible_components[component_type]

        param_def = {}
        for pname in possible_params:
            param_def[pname] = self.random_parameter_def(possible_params, pname)

        complementary_component = None
        keras_complementary_component = None

        if possible_complementary_components is not None and len(possible_complementary_components) > 0:
            c_def, c_params = possible_complementary_components[
                random.choice(list(possible_complementary_components))
            ]
            c_param_def = {}
            for pname in c_params:
                c_param_def[pname] = self.random_parameter_def(c_params, pname)
            complementary_component = [c_def, c_param_def]
            # we DON’T instantiate Keras layers here to keep this generator pure
            keras_complementary_component = None

        new_component = Component(
            representation=[component_def, param_def],
            keras_component=None,
            complementary_component=complementary_component,
            keras_complementary_component=keras_complementary_component,
            component_type=component_type,
        )
        return new_component

    def random_graph(self, node_range, node_content_generator, args=None):
        """
        Build a random DAG with 'node_range' nodes.
        """
        args = args or {}
        g = nx.DiGraph()

        for node in range(node_range):
            node_def = node_content_generator(**args)
            g.add_node(node, node_def=node_def)

            if node == 0:
                continue

            if node < node_range - 1 or node_range <= 2:
                precedent = random.randint(0, node - 1)
                g.add_edge(precedent, node)
            else:
                leaf_nodes = [n for n in g.nodes() if g.out_degree(n) == 0]
                root_node = min([n for n in g.nodes() if g.in_degree(n) == 0])
                if node in leaf_nodes:
                    leaf_nodes.remove(node)

                while len(leaf_nodes) > 0:
                    if len(leaf_nodes) <= 2:
                        ln = random.choice(leaf_nodes)
                        g.add_edge(ln, node)
                        leaf_nodes.remove(ln)
                    else:
                        leaf_nodes.append(root_node)
                        r1 = random.choice(leaf_nodes)
                        simple_path_nodes = [
                            n for path in nx.all_simple_paths(g, root_node, r1) for n in path
                        ]
                        leaf_nodes.remove(r1)
                        r2 = random.choice(leaf_nodes)
                        if (
                            g.in_degree(r2) >= 1
                            and r2 not in simple_path_nodes
                            and r2 != root_node
                        ):
                            g.add_edge(r1, r2)
                        leaf_nodes = [n for n in g.nodes() if g.out_degree(n) == 0]
                        if node in leaf_nodes:
                            leaf_nodes.remove(node)

        return g

    def random_module(
        self,
        global_configs,
        possible_components,
        possible_complementary_components,
        name=0,
        layer_type=ModuleComposition.INTERMED,
    ):
        """
        Sample a module by drawing a component graph.
        """
        node_range = self.random_parameter_def(global_configs, "component_range")
        logging.log(21, f"Generating {node_range} components")
        print(f"Generating {node_range} components")

        graph = self.random_graph(
            node_range=node_range,
            node_content_generator=self.random_component,
            args={
                "possible_components": possible_components,
                "possible_complementary_components": possible_complementary_components,
            },
        )

        self.save_graph_plot(f"module_{name}_{self.count}_module_internal_graph.png", graph)
        GraphOperator.count += 1

        new_module = Module(None, layer_type=layer_type, component_graph=graph)
        return new_module

    def random_blueprint(
        self,
        global_configs,
        possible_components,
        possible_complementary_components,
        input_configs,
        possible_inputs,
        possible_complementary_inputs,
        output_configs,
        possible_outputs,
        possible_complementary_outputs,
        input_shape,
        node_content_generator=None,
        args=None,
        name=0,
    ):
        """
        Sample a blueprint by composing (input | intermed* | output) modules.
        """
        args = args or {}
        node_range = self.random_parameter_def(global_configs, "module_range")
        logging.log(21, f"Generating {node_range} modules")
        print(f"Generating {node_range} modules")

        if node_content_generator is None:
            node_content_generator = self.random_module
            args = {
                "global_configs": global_configs,
                "possible_components": possible_components,
                "possible_complementary_components": possible_complementary_components,
            }

        # input module (single node)
        input_node = self.random_graph(
            node_range=1,
            node_content_generator=self.random_module,
            args={
                "global_configs": input_configs,
                "possible_components": possible_inputs,
                "possible_complementary_components": None,
                "layer_type": ModuleComposition.INPUT,
            },
        )

        # intermed modules
        intermed_graph = self.random_graph(
            node_range=node_range,
            node_content_generator=node_content_generator,
            args=args,
        )

        # output module (single node)
        output_node = self.random_graph(
            node_range=1,
            node_content_generator=self.random_module,
            args={
                "global_configs": output_configs,
                "possible_components": possible_outputs,
                "possible_complementary_components": possible_complementary_outputs,
                "layer_type": ModuleComposition.OUTPUT,
            },
        )

        graph = nx.union(input_node, intermed_graph, rename=("input-", "intermed-"))
        graph = nx.union(graph, output_node, rename=(None, "output-"))
        graph.add_edge("input-0", "intermed-0")
        graph.add_edge(f"intermed-{max(intermed_graph.nodes())}", "output-0")

        self.save_graph_plot(f"blueprint_{name}_module_level_graph.png", graph)
        new_blueprint = Blueprint(None, input_shape, module_graph=graph)
        return new_blueprint

    # ------------------------------------------------------------------ #
    # mutations
    # ------------------------------------------------------------------ #
    def mutate_by_node_removal(self, graph, generator_function, args=None):
        """
        Remove an internal node (not input/output), reconnect predecessors to successors.
        """
        args = args or {}
        new_graph = graph.copy()

        candidates = [
            n for n in new_graph.nodes() if new_graph.out_degree(n) > 0 and new_graph.in_degree(n) > 0
        ]
        if not candidates:
            return None

        selected = random.choice(candidates)

        preds = list(new_graph.predecessors(selected))
        succs = list(new_graph.successors(selected))
        if preds and succs:
            new_edges = [(p, s) for p in preds for s in succs]
            new_graph.remove_node(selected)
            new_graph.add_edges_from(new_edges)

        return new_graph

    def mutate_by_node_addition_in_edges(self, graph, generator_function, args=None):
        """
        Split a random existing edge by inserting a new node between (u -> v).
        """
        args = args or {}
        new_graph = graph.copy()

        # new node id (works for both int and 'intermed-i' naming)
        try:
            node_id = int(max(new_graph.nodes())) + 1
        except Exception:
            node_id = "intermed-" + str(
                max(
                    [int(str(n).split("-")[1]) for n in new_graph.nodes() if "input" not in str(n) and "output" not in str(n)]
                    + [0]
                )
                + 1
            )

        edges = list(new_graph.edges())
        if not edges:
            return None

        u, v = random.choice(edges)

        node_def = generator_function(**(args or {}))
        new_graph.add_node(node_id, node_def=node_def)

        new_graph.remove_edge(u, v)
        new_graph.add_edge(u, node_id)
        new_graph.add_edge(node_id, v)

        # (bug fix) previously referenced undefined 'leaf_nodes' here — removed.
        return new_graph

    def mutate_by_node_addition_outside_edges(self, graph, generator_function, args=None):
        """
        Insert a new node and connect it from a non-leaf predecessor to a successor
        that currently has in_degree == 1 and is not on the path from the root to the predecessor.
        If no such successor exists, split one of predecessor's outgoing edges.
        """
        args = args or {}
        new_graph = graph.copy()

        try:
            node_id = int(max(new_graph.nodes())) + 1
        except Exception:
            node_id = "intermed-" + str(
                max(
                    [int(str(n).split("-")[1]) for n in new_graph.nodes() if "input" not in str(n) and "output" not in str(n)]
                    + [0]
                )
                + 1
            )

        node_def = generator_function(**args)

        # choose a predecessor with at least one outgoing edge (not a leaf)
        predecessors = [n for n in new_graph.nodes() if new_graph.out_degree(n) > 0]
        if not predecessors:
            return None
        predecessor = random.choice(predecessors)

        # root for path calculations
        starting_node = min([n for n in new_graph.nodes() if new_graph.in_degree(n) == 0])
        simple_paths = [n for path in nx.all_simple_paths(new_graph, starting_node, predecessor) for n in path]

        # candidate successors: not inputs, in_degree == 1, and not on the predecessor’s path
        candidates = [n for n in new_graph.nodes() if new_graph.in_degree(n) == 1 and n not in simple_paths]

        new_graph.add_node(node_id, node_def=node_def)

        if not candidates:
            # fallback: split one outgoing edge of predecessor
            succ = random.choice(list(new_graph.successors(predecessor)))
            new_graph.remove_edge(predecessor, succ)
            new_graph.add_edge(predecessor, node_id)
            new_graph.add_edge(node_id, succ)
        else:
            successor = random.choice(candidates)
            new_graph.add_edge(predecessor, node_id)
            new_graph.add_edge(node_id, successor)

        # ensure there is a single leaf (the original code enforced it; keep the guard)
        leaf_nodes = [n for n in new_graph.nodes() if new_graph.out_degree(n) == 0]
        if len(leaf_nodes) != 1:
            return None

        return new_graph

    def mutate_by_node_replacement(self, graph, generator_function, args=None):
        """
        Replace the node_def of a random non-(input/output) node with a fresh one.
        """
        args = args or {}
        new_graph = graph.copy()

        candidates = [n for n in new_graph.nodes() if "input" not in str(n) and "output" not in str(n)]
        if not candidates:
            return None

        selected = random.choice(candidates)
        node_def = generator_function(**args)
        new_graph.nodes[selected]["node_def"] = node_def

        return new_graph

