from typing import List, cast

from dqo.relational.models import TableRef, ColumnRef, Condition, Const
from dqo.relational.query import Query
from dqo.relational.tree import RelationalTree
from dqo.relational.tree.node import SelectionNode, RelationNode, ProjectionNode, RelationColumn, JoinNode


def parse_relation_column(rc: RelationColumn) -> ColumnRef:
    t = TableRef(rc.relation.name, alias=rc.relation.alias)
    return ColumnRef(rc.column, table=t)


def parse_selection(selection: SelectionNode) -> List[Condition]:
    op = selection.operator

    conditions: List[Condition] = []
    if op.upper() in ['AND', 'OR']:
        for inner in selection.operands:
            conditions += parse_selection(cast(SelectionNode, inner))
    else:
        left, right = selection.operands

        col = None
        const = None

        ltr = True
        if isinstance(left, RelationColumn):
            col = parse_relation_column(left)
        else:
            const = left.text

        if isinstance(right, RelationColumn):
            ltr = False
            col = parse_relation_column(right)
        else:
            const = right.text

        if col is None or const is None:
            raise ValueError('Non relational selections are not supported')

        conditions = [Condition(col=col, value=Const(const), operator=op, ltr=ltr)]

    return conditions


def parse_tree(t: RelationalTree, keep_order=False) -> Query:
    q = Query(track_order=keep_order)
    for node in t.nodes():
        if isinstance(node, JoinNode):
            op = node.operator
            left, right = cast(List[RelationColumn], node.operands)
            q.add_join([parse_relation_column(left), parse_relation_column(right)])

        elif isinstance(node, SelectionNode):
            conditions = parse_selection(node)

            for cond in conditions:
                # FIXME : if not q.has_selection(cond):
                q.add_condition(cond)

        elif isinstance(node, RelationNode):
            tref = TableRef(name=node.name, alias=node.alias)
            q.add_table(tref, track=True)
        elif isinstance(node, ProjectionNode):
            for col in node.columns:
                # TODO: there are duplicate definitions of what a column is.
                #  need to clean this up.
                # TODO: also, binding a virtual schema to an actual database, causes type to become real.
                c = parse_relation_column(col)
                q.add_projection(c, func=col.func)

    return q
