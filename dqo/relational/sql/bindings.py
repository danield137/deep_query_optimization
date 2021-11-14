from __future__ import annotations

from typing import Optional

from dqo.db.models import Database
from dqo.relational import RelationalTree
from dqo.relational.tree.node import RelationColumn


def validate(tree: RelationalTree, db: Database) -> Optional[str]:
    for rel in tree.relations.values():
        if rel.name not in db:
            return f'Relation {rel.name} not found.'

    for selection in tree.get_selections():
        for col in selection.operands:
            if isinstance(col, RelationColumn):
                if col.column not in db[col.relation.name]:
                    return f'Bad selection: Column {col.column} not found in {col.relation.name}'

    for projection in tree.get_projections():
        for col in projection.columns:
            if col.column not in db[col.relation.name]:
                return f'Bad projection: Column {col.column} not found in {col.relation.name}'

    return None