
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'DQUOTES IN NAME QUOTES statement : NAME\n                | IN\n                | NAME IN\n                | QUOTES statement QUOTES\n    '
    
_lr_action_items = {'NAME':([0,4,],[2,2,]),'IN':([0,2,4,],[3,5,3,]),'QUOTES':([0,2,3,4,5,6,7,],[4,-1,-2,4,-3,7,-4,]),'$end':([1,2,3,5,7,],[0,-1,-2,-3,-4,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'statement':([0,4,],[1,6,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> statement","S'",1,None,None,None),
  ('statement -> NAME','statement',1,'p_statement','test_me.py',46),
  ('statement -> IN','statement',1,'p_statement','test_me.py',47),
  ('statement -> NAME IN','statement',2,'p_statement','test_me.py',48),
  ('statement -> QUOTES statement QUOTES','statement',3,'p_statement','test_me.py',49),
]
