from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import dacite
from dacite import Config


@dataclass
class Serializable:
    @classmethod
    def copy(cls, other):
        return cls.from_dict(other)

    @classmethod
    def load(cls, file):
        with open(file, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save(self, path: str):
        basepath = os.path.dirname(path)
        os.makedirs(basepath, exist_ok=True)
        with open(path, 'w+') as out:
            d = asdict(self)
            json.dump(d, out)

    @classmethod
    def from_dict(cls, data):
        obj = dacite.from_dict(cls, data, config=Config(
            cast=cls.get_casts(),
            # this is due to a bug with optionals
            # retry with higher python (3.9)
            check_types=False
        ))

        if hasattr(obj, '__post_init__'):
            post_init = getattr(obj, '__post_init__')
            post_init()

        return obj

    @classmethod
    def get_casts(cls):
        return []

    @classmethod
    def get_forward_refs(cls):
        return None

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, default=str)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str)
        return cls.from_dict(d)
