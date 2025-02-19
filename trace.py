#!/usr/bin/env python
import os
import ast
import networkx as nx
import matplotlib.pyplot as plt

def get_key(filename, qualified_name):
    """Build a unique key for a function defined in a file."""
    return f"{filename}::{qualified_name}"

# Global dictionary to hold discovered functions.
# Keys are strings like "filename.py::QualifiedName" and values are dicts with metadata and calls.
functions = {}

class CallGraphVisitor(ast.NodeVisitor):
    """
    This visitor records function definitions and function calls.
    It uses stacks to keep track of the current class and function.
    """
    def __init__(self, filename):
        self.filename = filename
        self.current_function = []  # stack of function names (qualified names)
        self.current_class = []     # stack of class names

    def visit_FunctionDef(self, node):
        # Create a qualified name: if inside a class, include the class name.
        if self.current_class:
            qualified_name = f"{self.current_class[-1]}.{node.name}"
        else:
            qualified_name = node.name
        key = get_key(self.filename, qualified_name)
        if key not in functions:
            functions[key] = {
                'filename': self.filename,
                'lineno': node.lineno,
                'name': qualified_name,
                'calls': set()
            }
        # Push this function onto the stack.
        self.current_function.append(qualified_name)
        self.generic_visit(node)
        self.current_function.pop()

    def visit_ClassDef(self, node):
        # Push class name and process its body.
        self.current_class.append(node.name)
        self.generic_visit(node)
        self.current_class.pop()

    def visit_Call(self, node):
        # Identify which function (or module-level code) this call is in.
        if self.current_function:
            if self.current_class:
                caller_qualified = f"{self.current_class[-1]}.{self.current_function[-1]}"
            else:
                caller_qualified = self.current_function[-1]
        else:
            caller_qualified = "<module>"
        caller_key = get_key(self.filename, caller_qualified)
        if caller_key not in functions:
            functions[caller_key] = {
                'filename': self.filename,
                'lineno': 0,
                'name': caller_qualified,
                'calls': set()
            }
        # Resolve the callee and record the call if possible.
        callee_key = self.resolve_call(node)
        if callee_key:
            functions[caller_key]['calls'].add(callee_key)
        self.generic_visit(node)

    def resolve_call(self, node):
        """
        Attempt to extract a candidate callee key from a call node.
        Handles:
          - Simple calls: foo()
          - Method calls: self.method()
          - Module calls: module.func()
        Returns a key if one can be formed.
        """
        if isinstance(node.func, ast.Name):
            callee_qualified = node.func.id
            return get_key(self.filename, callee_qualified)
        elif isinstance(node.func, ast.Attribute):
            # Handle self.method() calls.
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                if self.current_class:
                    callee_qualified = f"{self.current_class[-1]}.{node.func.attr}"
                    return get_key(self.filename, callee_qualified)
            # Handle calls like module.func() (heuristic: assume local file).
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                mod_filename = module_name + ".py"
                callee_qualified = node.func.attr
                return get_key(mod_filename, callee_qualified)
        return None

def process_file(filepath):
    """Parse a Python file and run the AST visitor on it."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return

    visitor = CallGraphVisitor(filepath)
    # Create a pseudo-function for module-level code.
    module_key = get_key(filepath, "<module>")
    if module_key not in functions:
        functions[module_key] = {
            'filename': filepath,
            'lineno': 0,
            'name': "<module>",
            'calls': set()
        }
    visitor.current_function.append("<module>")
    visitor.visit(tree)
    visitor.current_function.pop()

def build_graph():
    """Build a directed graph from the functions dictionary."""
    G = nx.DiGraph()
    for key, data in functions.items():
        # Label: function name and its location.
        label = f"{data['name']}\n{os.path.basename(data['filename'])}:{data['lineno']}"
        G.add_node(key, label=label)
    for key, data in functions.items():
        for callee_key in data['calls']:
            if callee_key in functions:
                G.add_edge(key, callee_key)
    return G

def get_reachable_nodes(start_key, graph):
    """Return the set of nodes reachable from start_key (using DFS)."""
    reachable = set()
    stack = [start_key]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for neighbor in graph.successors(node):
            if neighbor not in reachable:
                stack.append(neighbor)
    return reachable

def main():
    # Process all Python files in the current folder.
    for filename in os.listdir("."):
        if filename.endswith(".py"):
            process_file(filename)

    # Build the full call graph.
    G = build_graph()

    # Use the main file's module-level code as the entry point.
    main_entry = get_key("main_trajectory_calvin.py", "<module>")
    if main_entry not in G:
        print("Entry point for main_trajectory_calvin.py not found in the call graph.")
        return

    # Filter the graph to include only nodes reachable from the main file.
    reachable = get_reachable_nodes(main_entry, G)
    subG = G.subgraph(reachable).copy()

    # Create a layout and draw the graph.
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subG, k=0.5, iterations=50)
    labels = nx.get_node_attributes(subG, 'label')
    nx.draw(
        subG, pos,
        with_labels=True,
        labels=labels,
        node_color='skyblue',
        edge_color='gray',
        node_size=2000,
        font_size=8,
        arrowsize=20
    )
    plt.title("Static Call Graph (Local Functions Only)")
    plt.tight_layout()

    # Save the graph as an image file instead of showing it.
    output_filename = "static_call_graph.png"
    plt.savefig(output_filename)
    print(f"Call graph image saved as {output_filename}")

if __name__ == "__main__":
    main()
