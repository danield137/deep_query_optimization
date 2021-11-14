"""
This is a syntax parser based on PLY
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import ply.lex as lex
import ply.yacc as yacc

logger = logging.getLogger('relational.sql.ast')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

operators = ['GT', 'LT', 'EQ', 'MINUS', 'PLUS', 'PRODUCT', 'DIV']
syntax = ['QTEXT', 'LP', 'RP', 'NAME', 'COMMA', 'DOT', 'NUMBER', 'QUOTES', 'DQUOTES', 'EXLM']

reserved = {
    'select': 'SELECT',
    'from': 'FROM',
    'where': 'WHERE',
    'order': 'ORDER',
    'group': 'GROUP',
    'by': 'BY',
    'is': 'IS',
    'limit': 'LIMIT',
    'as': 'AS',
    'asc': 'ASC',
    'desc': 'DESC',
    'when': 'WHEN',
    'case': 'CASE',
    'then': 'THEN',
    'end': 'END',
    'having': 'HAVING',
    'avg': 'AVG',
    'max': 'MAX',
    'min': 'MIN',
    'sum': 'SUM',
    'count': 'COUNT',
    'and': 'AND',
    'or': 'OR',
    'between': 'BETWEEN',
    'like': 'LIKE',
    'in': 'IN',
    'not': 'NOT',
    'null': 'NULL',
    'true': 'TRUE',
    'false': 'FALSE'
}

tokens = syntax + operators + list(reserved.values())


class ASTNode:
    parent: ASTNode
    children: List[ASTNode]

    def __init__(self, name: str):
        self.name = name
        self.children = []

    def add(self, *args: ASTNode):
        for item in args:
            item.parent = self
            self.children.append(item)

        return self

    def pprint(self):
        setattr(self, 'depth', 0)
        nodes = [self]
        while nodes:
            node = nodes.pop(0)

            depth = getattr(node, 'depth', 0)
            if type(node) is ASTNode:

                for c in node.children:
                    setattr(c, 'depth', getattr(node, 'depth', 0) + 1)
                nodes = [*node.children, *nodes]

            print('  ' * depth, node)

    def __repr__(self):
        return self.name


def t_QTEXT(t):
    r'(\"|\')[a-zA-Z0-9_\- \[\]\{\}\(\)\!\@\#\$\%\^\&\*\,\`\<\>\\\/\|\+\.\:\;]*(\"|\')'
    return t


def t_GT(t):
    r'\>'
    return t


def t_LT(t):
    r'\<'
    return t


def t_DQUOTES(t):
    r'\"'
    return t


def t_QUOTES(t):
    r'\''
    return t


def t_LP(t):
    r'\('
    return t


def t_DOT(t):
    r'\.'
    return t


def t_RP(t):
    r'\)'
    return t


def t_COMMA(t):
    r','
    return t


def t_NAME(t):
    r'(?i)[a-zA-Z_]+[a-zA-Z0-9_]*\.?[a-zA-Z0-9_]*'
    l = str(t.value).lower()
    if l in reserved:
        t.type = reserved[l]

    return t


t_EXLM = r'!'
t_MINUS = r'\-'
t_DIV = r'\/'
t_PLUS = r'\+'
t_EQ = r'\='
t_PRODUCT = r'\*'

t_NUMBER = r'\d+'

# IGNORED
t_ignore = " \t\r\n"


def t_error(t):
    logger.error("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# LEX ANALYSIS
lex.lex()


# PARSING
def p_query(t):
    '''query :  select
            | LP query RP
                '''
    if len(t) == 2:
        t[0] = t[1]
    else:
        t[0] = t[2]


def p_select(t):
    '''select :   SELECT list FROM table WHERE predicates GROUP BY list HAVING predicates ORDER BY list LIMIT NUMBER
                | SELECT list FROM table WHERE predicates GROUP BY list HAVING predicates ORDER BY list
                | SELECT list FROM table WHERE predicates GROUP BY list ORDER BY list LIMIT NUMBER
                | SELECT list FROM table WHERE predicates GROUP BY list ORDER BY list
                | SELECT list FROM table WHERE predicates GROUP BY list HAVING predicates LIMIT NUMBER
                | SELECT list FROM table WHERE predicates GROUP BY list HAVING predicates
                | SELECT list FROM table WHERE predicates GROUP BY list LIMIT NUMBER
                | SELECT list FROM table WHERE predicates ORDER BY list LIMIT NUMBER
                | SELECT list FROM table WHERE predicates ORDER BY list
                | SELECT list FROM table WHERE predicates GROUP BY list
                | SELECT list FROM table WHERE predicates LIMIT NUMBER
                | SELECT list FROM table WHERE predicates
                | SELECT list FROM table GROUP BY list HAVING predicates LIMIT NUMBER
                | SELECT list FROM table GROUP BY list LIMIT NUMBER
                | SELECT list FROM table ORDER BY list LIMIT NUMBER
                | SELECT list FROM table GROUP BY list HAVING predicates
                | SELECT list FROM table GROUP BY list
                | SELECT list FROM table ORDER BY list
                | SELECT list FROM table LIMIT NUMBER
                | SELECT list FROM table '''
    t[0] = ASTNode('[QUERY]').add(
        ASTNode('[SELECT]').add(t[2]),
        ASTNode('[FROM]').add(t[4])
    )

    index = 5

    while index < len(t):
        if type(t[index]) is str:
            tu = t[index].upper()
            if tu == 'WHERE':
                t[0].add(ASTNode('[WHERE]').add(t[index + 1]))

                index += 2
            elif tu == 'GROUP' or tu == 'ORDER':
                t[0].add(ASTNode(f'[{t[index]} BY]').add(t[index + 2]))

                if len(t) > index + 3:
                    if type(t[index + 3]) is str and t[index + 3].upper() == 'HAVING':
                        t[0].add(ASTNode('[HAVING]').add(t[index + 4]))

                        index += 2

                index += 3
            elif tu == 'LIMIT':
                t[0].add(ASTNode('[LIMIT]').add(ASTNode(t[index + 1])))
                # limit is the final char
                break


def p_table(t):
    '''table : NAME
            | LP query RP
            | NAME AS NAME
            | table AS NAME
            | table COMMA table'''
    if len(t) == 2:
        t[0] = ASTNode('[TABLE]').add(ASTNode(t[1]))
    elif type(t[2]) is str and t[2].upper() == 'AS':
        if isinstance(t[1], ASTNode):
            if t[1].children[0].name == '[QUERY]':
                t[0] = ASTNode('[TABLE]').add(t[1].children[0], ASTNode('AS'), ASTNode(t[3]))
            else:
                t[0] = ASTNode('[ALIAS]').add(t[1], ASTNode('AS'), ASTNode(t[3]))
        else:
            t[0] = ASTNode('[TABLE]').add(ASTNode(t[1]), ASTNode('AS'), ASTNode(t[3]))
    elif t[2] == ',':
        children = []
        for i in [1, 3]:
            if t[i].name == '[TABLE]' or t[i] == '[ALIAS]':
                children.append(t[i])
            else:
                children += t[i].children
        t[0] = ASTNode('[TABLES]').add(*children)
    else:
        t[0] = ASTNode('[TABLE]').add(t[2])


def p_predicates(t):
    ''' predicates  : condition
             | LP condition RP
             | LP predicates AND predicates RP
             | LP predicates OR predicates RP
             | predicates AND predicates
             | predicates OR predicates
              '''

    nodes = t[1:-1] if t[1] == '(' else t
    if len(nodes) == 2:
        t[0] = nodes[1]
    elif type(nodes[2]) is str:
        tu = nodes[2].upper()
        if tu == ',':
            t[0] = ASTNode('[CONDITIONS]').add(nodes[1], nodes[3])
        elif tu == 'AND':
            t[0] = ASTNode('[CONDITIONS]').add(nodes[1], ASTNode('[AND]'), nodes[3])
        elif tu == 'OR':
            t[0] = ASTNode('[CONDITIONS]').add(nodes[1], ASTNode('[OR]'), nodes[3])
        elif tu == 'BETWEEN':
            temp = '%s >= %s & %s <= %s' % (nodes[1], str(nodes[3]), nodes[1], str(nodes[5]))
            t[0] = ASTNode('[CONDITION]').add(ASTNode('[TERM]'), ASTNode(temp))
        elif tu == 'IN':
            t[0] = ASTNode('[CONDITION]').add(ASTNode(nodes[1]), ASTNode('[IN]'), nodes[4])
        elif tu == '<' and len(t) == 4:
            temp = '%s < %s' % (str(t[1]), str(t[3]))
            t[0] = ASTNode('[CONDITION]').add(ASTNode('[TERM]'), ASTNode(temp))
        elif tu == '=' and len(nodes) == 4:
            temp = '%s = %s' % (str(nodes[1]), str(nodes[3]))

            t[0] = ASTNode('[CONDITION]').add(ASTNode('[TERM]'), ASTNode(temp))
        elif tu == '>' and len(nodes) == 4:
            temp = '%s > %s' % (str(nodes[1]), str(nodes[3]))
            t[0] = ASTNode('[CONDITION]').add(ASTNode('[TERM]'), ASTNode(temp))
    else:
        t[0] = ASTNode('')


def p_numeric(t):
    ''' numeric : NUMBER
        | NUMBER DOT NUMBER
        | numeric PLUS numeric
        | numeric PRODUCT numeric
        | numeric DIV numeric
        | numeric MINUS numeric
        | LP numeric RP
    '''
    t[0] = ''.join(t[1:])


def p_condition(t):
    ''' condition : NAME operator numeric
                  | numeric operator NAME
                  | numeric operator numeric
                  | NAME operator agg
                  | NAME operator NAME
                  | NAME operator QTEXT
                  | QTEXT operator NAME
                  | list operator list
                  | list operator NUMBER
                  | NAME LIKE QTEXT
                  | NAME NOT LIKE QTEXT
                  | NAME IS NULL
                  | NAME IS NOT NULL
                  | NAME IS TRUE
                  | NAME IS FALSE
                  | NAME EQ TRUE
                  | NAME EQ FALSE
                  | NAME EXLM EQ TRUE
                  | NAME EXLM EQ FALSE
                  | NAME IS NOT TRUE
                  | NAME IS NOT FALSE
                  | NAME BETWEEN NUMBER AND NUMBER
                  | NAME BETWEEN QTEXT AND QTEXT
                  | NAME IN LP query RP
                  | NAME IN LP list RP
                  '''
    t[0] = ASTNode('[CONDITION]')
    i = 1
    if str(t[2]).upper() == 'IN':
        values_node = ASTNode('[VALUES]')
        if str(t[4]) == '[FIELDS]':
            for f in t[4].children:
                values_node.add(f.children[0])
        else:
            values_node.add(t[4].children[0] if type(t[4].children[0]) is ASTNode else ASTNode(t[4].children[0]))
        t[0].add(
            ASTNode(t[1]),
            ASTNode('[OP]').add(ASTNode('IN')),
            values_node
        )
    elif t[2] == 'IS':
        t[0].add(ASTNode(t[1]),
                 ASTNode('[OP]').add(ASTNode(' '.join(t[2:-1]))),
                 ASTNode(t[:][-1])
                 )
    elif t[2] == 'BETWEEN':
        t[0].add(ASTNode(t[1]),
                 ASTNode('[OP]').add(ASTNode('BETWEEN')),
                 ASTNode('[VALUES]').add(ASTNode(t[3]), ASTNode(t[5]))
                 )
    elif str(t[2]) == 'LIKE':
        t[0].add(ASTNode(t[1]),
                 ASTNode('[OP]').add(ASTNode('LIKE')),
                 ASTNode(t[3])
                 )
    elif str(t[2]) == 'NOT':
        if str(t[3]) == 'LIKE':
            t[0].add(ASTNode(t[1]),
                     ASTNode('[OP]').add(ASTNode('NOT LIKE')),
                     ASTNode(t[4]))
        else:
            t[0].add(ASTNode(t[1]),
                     ASTNode('[OP]').add(ASTNode(t[2])),
                     ASTNode(t[3]))
    elif str(t[2]) == '[OP]':
        t[0].add(ASTNode(t[1]),
                 t[2],
                 ASTNode(t[3])
                 )
    elif len(t) == 4:
        t[0].add(ASTNode(t[1]),
                 ASTNode('[OP]').add(ASTNode(t[2])),
                 ASTNode(t[3])
                 )
    else:
        raise ValueError('huh?', list(t))


def p_operator(t):
    ''' operator : GT
                | LT
                | EQ
                | GT EQ
                | LT EQ
                | EXLM EQ
    '''
    if len(t) == 2:
        t[0] = ASTNode('[OP]').add(ASTNode(t[1]))
    elif t[1] == '<' and t[2] == '=':
        t[0] = ASTNode('[OP]').add(ASTNode('<='))
    elif t[1] == '>' and t[2] == '=':
        t[0] = ASTNode('[OP]').add(ASTNode('>='))
    elif t[1] == '!' and t[2] == '=':
        t[0] = ASTNode('[OP]').add(ASTNode('!='))
    else:
        raise ValueError('huh?', list(t))


def p_agg(t):
    ''' agg : SUM LP NAME RP
            | AVG LP NAME RP
            | COUNT LP NAME RP
            | MIN LP NAME RP
            | MAX LP NAME RP
            | COUNT LP PRODUCT RP
            | agg AS NAME
            '''
    if len(t) == 4 and str(t[2]).upper() == 'AS':
        t[0] = t[1].add(ASTNode('AS'), ASTNode(t[3]))
    else:
        t[0] = ASTNode('[FIELD_AGG]').add(
            ASTNode('[AGG]').add(
                ASTNode(str(t[1]))
            ),
            ASTNode(str(t[3]))
        )


def p_list(t):
    ''' list : PRODUCT
             | NAME
             | NUMBER
             | NUMBER COMMA NUMBER
             | NAME COMMA NAME
             | QTEXT
             | DQUOTES list DQUOTES
             | QUOTES list QUOTES
             | NAME AS NAME
             | NAME DOT NAME
             | list COMMA list
             | list AND NAME
             | list OR NAME
             | agg
             '''
    if len(t) == 2:
        if type(t[1]) is ASTNode:
            t[0] = t[1]
        else:
            t[0] = ASTNode('[FIELD]').add(ASTNode(t[1]))
    elif t[1] == '"':
        t[0] = ASTNode('[VALUE]')
        t[0].add(ASTNode('"' + t[2] + '"'))
    elif t[2] == ',':
        t[0] = ASTNode('[FIELDS]')

        for i in [1, 3]:
            if str(t[i]) in ('[FIELDS]'):
                for v in t[i].children:
                    t[0].add(v)
            elif str(t[i]) in ('[FIELD]', '[FIELD_AGG]'):
                t[0].add(t[i])
            else:
                t[0].add(ASTNode('[FIELD]').add(ASTNode(t[i])))

    elif type(t[2]) is str and t[2].upper() == 'AS':
        t[0] = ASTNode('[FIELD]')
        t[0].add(ASTNode(t[1]), ASTNode(t[2]), ASTNode(t[3]))
    else:
        temp = f'{t[1]}.{t[3]}'
        t[0] = ASTNode('[FIELD]')
        t[0].add(ASTNode(temp))


def p_error(t):
    logger.error("Syntax error at '%s'" % t.value)


yacc.yacc()


@dataclass
class AbstractSyntaxTree:
    """
    Abstract syntax tree based on ply.py
    Parses SQL to tokens which are then represented as as tree for further processing.
    """
    query: str
    root: ASTNode

    @staticmethod
    def parse(query: str) -> AbstractSyntaxTree:
        clean = query.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
        ast = AbstractSyntaxTree(
            root=yacc.parse(clean),
            query=query)
        return ast
