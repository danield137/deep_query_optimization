from dqo.relational.query import Query
from dqo.relational.query.parser import parse_tree
from dqo.relational.sql.ast import AbstractSyntaxTree
from dqo.relational.tree import RelationalTree
from dqo.relational.tree.parser import parse_ast


class SQLParser:
    @staticmethod
    def validate(sql: str, silent=True) -> bool:
        try:
            import logging

            ast_logger = logging.getLogger('relational.sql.ast')
            old_level = ast_logger.level
            ast_logger.setLevel(logging.CRITICAL)
            success = SQLParser.to_relational_tree(sql) is not None
            ast_logger.setLevel(old_level)

            return success
        except:
            return False

    @staticmethod
    def to_relational_tree(sql: str) -> RelationalTree:
        return parse_ast(AbstractSyntaxTree.parse(sql).root, sql)

    @staticmethod
    def to_ast(sql: str) -> AbstractSyntaxTree:
        return AbstractSyntaxTree.parse(sql)

    @staticmethod
    def to_query(sql: str) -> Query:
        return parse_tree(parse_ast(AbstractSyntaxTree.parse(sql).root, sql))
