"""Minimal Newick parsing and rectangular layout for the visualization notebook.

DAMICORE's own dependency set has no plotting or tree library (`tests/architecture` closes
it), so this stays stdlib-only and produces plain (x, y) line segments; the notebook is the
only place that touches matplotlib. Not a general Newick library -- it accepts exactly what
`damicore_tree_builder` emits: single-quoted labels, `:`-branch lengths, and a trailing `;`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    name: str | None
    length: float
    children: list[Node] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class NewickParseError(ValueError):
    """Raised when a Newick string does not match the subset this parser accepts."""


def parse_newick(text: str) -> Node:
    """Parse a single-tree Newick string into a `Node` tree.

    Raises
    ------
    NewickParseError
        The string is empty, unterminated, or does not end with `;`.
    """
    stripped = text.strip()
    if not stripped.endswith(";"):
        raise NewickParseError("Newick string must end with ';'")
    body = stripped[:-1]
    position = 0

    def peek() -> str | None:
        return body[position] if position < len(body) else None

    def parse_label() -> str | None:
        nonlocal position
        if peek() == "'":
            position += 1
            start = position
            while peek() != "'":
                if position >= len(body):
                    raise NewickParseError("Unterminated quoted label")
                position += 1
            label = body[start:position]
            position += 1
            return label
        start = position
        while peek() is not None and peek() not in ",:()":
            position += 1
        return body[start:position] or None

    def parse_length() -> float:
        nonlocal position
        if peek() != ":":
            return 0.0
        position += 1
        start = position
        while peek() is not None and peek() not in ",()":
            position += 1
        return float(body[start:position])

    def parse_node() -> Node:
        nonlocal position
        children: list[Node] = []
        if peek() == "(":
            position += 1
            children.append(parse_node())
            while peek() == ",":
                position += 1
                children.append(parse_node())
            if peek() != ")":
                raise NewickParseError("Unbalanced parentheses")
            position += 1
        name = parse_label()
        length = parse_length()
        return Node(name=name, length=length, children=children)

    root = parse_node()
    if position != len(body):
        raise NewickParseError("Trailing characters after root node")
    return root


@dataclass
class Placed:
    node: Node
    x: float
    y: float


def layout(root: Node) -> tuple[list[Placed], list[tuple[Placed, Placed]]]:
    """Assign plot coordinates: x from cumulative branch length, y from leaf order.

    Returns every placed node and the parent-child edges connecting them, ready to draw as
    matplotlib line segments (one horizontal segment per edge; the caller adds the vertical
    connectors between a node's children).
    """
    placed: list[Placed] = []
    edges: list[tuple[Placed, Placed]] = []
    next_leaf_y = 0.0

    def place(node: Node, x: float) -> Placed:
        nonlocal next_leaf_y
        if node.is_leaf:
            here = Placed(node, x, next_leaf_y)
            next_leaf_y += 1.0
            placed.append(here)
            return here
        child_points = [place(child, x + child.length) for child in node.children]
        y = sum(point.y for point in child_points) / len(child_points)
        here = Placed(node, x, y)
        placed.append(here)
        for point in child_points:
            edges.append((here, point))
        return here

    place(root, 0.0)
    return placed, edges
