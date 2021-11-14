from __future__ import annotations

from typing import Dict, cast, Tuple

from dqo.relational.sql.ast import ASTNode, AbstractSyntaxTree
from dqo.relational.tree import RelationalTree
from dqo.relational.tree.node import RelationColumn, RelationNode, AliasNode, SelectionNode, JoinNode, ProjectionNode, \
    OrNode, Operand, MultiValueOperand


def parse_sql(sql: str) -> RelationalTree:
    ast = AbstractSyntaxTree.parse(sql.strip())
    result = parse_ast(ast.root, ast.query)
    return result


def parse_ast(root: ASTNode, sql: str = None, tree: RelationalTree = None) -> RelationalTree:
    if tree is None:
        tree = RelationalTree(sql)

    parts = {c.name.replace('[', '').replace(']', '').upper(): c for c in root.children}

    has_from = 'FROM' in parts
    has_where = 'WHERE' in parts

    if has_from:
        tree.relations = parse_from_clause(parts['FROM'])

    top_level = parse_select_clause(parts['SELECT'], tree)

    if tree.root is None:
        tree.root = top_level
        top_level.out = True

    if has_where:
        parse_where_clause(parts['WHERE'], tree)

    # TODO: handle this better
    # this can happen when the projection part doesn't include all relations.
    # in that case, the join clause might get wrapped above the root projection.
    # it is no necessarily wrong to do this, and has three possible fixes:
    # 1. join logic should keep track of tree root, and always inject under it.
    #    this only happens when joins are used as selections.
    #    in such cases it does not matter which nodes is first.
    #    it MIGHT matter if this tree is a subtree passed out, and projections are to be bubbled up.
    #    as of the time writing this, I do not ensure projection propagation.
    # 2. Propagate projections inside the current tree - i.e. make sure all projections are bubbled up to the root.
    # 3. Bubble up the root - hacky, but works
    if tree.root.parent is not None:
        tree.root.bubble_up()
    return tree


def parse_select_clause(select_clause: ASTNode, tree) -> ProjectionNode:
    if select_clause.children[0].name in ('[FIELD]', '[FIELD_AGG]'):
        fields = [parse_field(select_clause.children[0], tree)]
    elif select_clause.children[0].name == '[FIELDS]':
        fields = [parse_field(f, tree) for f in select_clause.children[0].children]
    else:
        raise ValueError('huh?', select_clause)

    if len(fields) == 1 and fields[0].text == '*':
        return ProjectionNode(columns=[], wildcard=True)
    else:
        return ProjectionNode(columns=[f for f in fields if isinstance(f, RelationColumn)])


def parse_from_clause(from_clause: ASTNode) -> Dict[str, RelationNode]:
    relations: Dict[str, RelationNode] = {}

    if from_clause.children[0].name == '[TABLE]':
        tables = [from_clause.children[0]]
    elif from_clause.children[0].name == '[TABLES]':
        tables = from_clause.children[0].children
    else:
        raise ValueError('huh?', from_clause)

    for t in tables:
        if t.name in ['[TABLE]', '[ALIAS]']:
            # TODO: this bit can be done smarter
            alias = str(t.children[2]) if len(t.children) == 3 else None
            k, v = parse_table(t, alias)
        else:
            raise Exception()

        relations[k] = v
    return relations


def parse_where_clause(where_clause: ASTNode, tree: RelationalTree) -> SelectionNode:
    return parse_condition(where_clause.children[0], tree)


def parse_table(table, alias=None) -> Tuple[str, RelationNode]:
    if table.children[0].name == '[QUERY]':
        # TODO: it would be nice to pass the actual sub-select for debugging purposes
        #  the problem is it requires the ast tree to either hold or track the exact slice or slice index
        inner_tree = parse_ast(table.children[0])
        an = AliasNode(alias=alias or table.children[0].name, relations=list(inner_tree.relations.values()))
        inner_tree.root.set_parent(an)
        return an.alias, an
    elif len(table.children) == 1:
        return table.children[0].name, RelationNode(name=table.children[0].name)
    elif table.children[1].name == 'AS':
        return table.children[2].name, RelationNode(
            name=table.children[0].name,
            alias=table.children[2].name)
    else:
        raise ValueError('huh?', table.children)


def parse_operand(operand: ASTNode, tree: RelationalTree) -> Operand:
    if operand.name == '[QUERY]':
        pass
    else:
        if '"' in operand.name or "'" in operand.name:
            return Operand(operand.name)
        elif operand.name.upper() in ('TRUE', 'FALSE'):
            return Operand(operand.name.upper() == 'TRUE')
        elif operand.name.upper() == 'NULL':
            return Operand('NULL')
        elif operand.name == '[VALUES]':
            return MultiValueOperand(values=[c.name for c in operand.children])
        elif '.' in operand.name:
            relation_name, column_name = operand.name.split('.')
            relation = tree.relations[relation_name]

            return RelationColumn(operand.name, relation, column_name)
        else:
            try:
                if isinstance(eval(operand.name), (float, int)):
                    return Operand(operand.name)
            except:
                pass

            relation = list(tree.relations.values())[0]

            return RelationColumn(operand.name, relation, operand.name)


def parse_condition(condition: ASTNode, tree: RelationalTree) -> SelectionNode:
    left, operator, right = condition.children

    if condition.name == '[CONDITIONS]':
        operands = [parse_condition(left, tree), parse_condition(right, tree)]
        if operator.name == '[AND]':
            # return the highest selection node
            return operands[0].intersection(operands[1])
        elif operator.name == '[OR]':
            # Or nodes are special.
            # If the Or condition is based on two relations, it should be above a join
            # otherwise, it doesn't really matter
            left = cast(RelationNode, operands[0])
            right = cast(RelationNode, operands[1])
            # relations = set(
            #     list(left.relations()) +
            #     list(right.relations())
            # )

            or_node = OrNode(left, right)

            intersection = left.intersection(right, exclude_self=True)
            if isinstance(intersection, JoinNode):
                join = intersection
            else:
                join = next((d for d in intersection.descendents() if isinstance(d, JoinNode)), None)

            if join:
                left.detach()
                right.detach()
                # if there is a join, we need to push the or node above it
                # TODO: should make sure the join here matches the relations from above

                or_node.push_above(join)
            elif intersection:
                # if there is no join but there is an intersection - this means that we still didn't get a chance
                # to process the join. need to be careful here - should add this Or but ensure that join logic knows
                # to push itself underneath it
                left.detach()
                right.detach()

                or_node.push_under(intersection)
            else:
                # this mean this are two separate parts of the tree
                # either because there is a join down the line, or maybe a syntax error
                parent = left.parent or right.parent  # Shouldn't matter which side if both are actual nodes
                children = list(set(left.detach() + right.detach()))

                or_node.set_parent(parent)
                or_node.set_children(children)

            return or_node
        else:
            raise ValueError('hu?', operator)

    elif condition.name == '[CONDITION]':
        operands = [parse_operand(left, tree), parse_operand(right, tree)]
        relations = []

        for operand in operands:
            if isinstance(operand, RelationColumn):
                relations.append(operand.relation)

        if len(relations) == 2:
            sn = JoinNode(operator=str(operator.children[0]), operands=operands, left=relations[0], right=relations[1])
        else:
            sn = SelectionNode(operator=operator.children[0].name, operands=operands)
            if len(relations) == 1:
                sn.push_above(relations[0])

        return sn
    else:
        raise ValueError(condition)


def parse_field(field: ASTNode, tree: RelationalTree) -> Operand:
    if field.children[0].name == "*":
        return Operand(text='*')
    elif field.name == '[FIELD]':
        column_name = field.children[0].name
        alias = field.children[2].name if len(field.children) == 3 else column_name
        if column_name and '.' in column_name:
            relation_name, column_name = column_name.split('.')
            relation = tree.relations[relation_name]
            return RelationColumn(relation=relation, column=column_name, alias=alias, text=field.children[0].name)
        else:
            # TODO: this is probably wrong.
            #  should find the actual relation related to this field instead od assuming it's the first one
            return RelationColumn(relation=list(tree.relations.values())[0], column=column_name, alias=alias, text=field.children[0].name)
    elif field.name == '[FIELD_AGG]':
        # TODO: handle aggregations properly
        agg_func = field.children[0].children[0].name
        column_name = field.children[1].name
        alias = field.children[3].name if len(field.children) == 4 else column_name

        text = f'{agg_func}({alias})'
        if column_name and '.' in column_name:
            relation_name, column_name = column_name.split('.')
            relation = tree.relations[relation_name]
            pn = RelationColumn(
                relation=relation, column=column_name, alias=alias, text=text, aggregated=True, func=agg_func
            )
        else:
            pn = RelationColumn(
                relation=list(tree.relations.values())[0], column=column_name, alias=alias, text=text, aggregated=True,
                func=agg_func
            )

        return pn
    else:
        raise ValueError('huh?', field)
