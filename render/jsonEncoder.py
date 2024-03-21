import json

class CompactListEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        if isinstance(o, list) and all(isinstance(i, list) and len(i) == 2 for i in o):
            yield '[' + ', '.join(map(str, o)) + ']'
        else:
            yield from super().iterencode(o, _one_shot=_one_shot)
