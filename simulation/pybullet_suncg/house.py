import json
import numpy as np


class Level:
    def __init__(self, dict_, house):
        self.__dict__ = dict_
        self.house = house
        invalid_nodes = [n['id'] for n in self.nodes if (not n['valid']) and 'id' in n]
        self.nodes = [Node(n,self) for n in self.nodes if n['valid']]
        self.node_dict = {n.id: n for n in self.nodes}
        self.nodes = self.node_dict.values()  # deduplicate nodes with same id
        self.rooms = [Room(n, ([self.node_dict[i] for i in [f'{self.id}_{j}' \
                      for j in list(set(n.nodeIndices))] if i not in invalid_nodes]), self) \
                      for n in self.nodes if n.type == 'Room' and hasattr(n, 'nodeIndices')]


class Room:
    def __init__(self, room, nodes, level):
        self.__dict__ = room.__dict__
        self.nodes = nodes
        self.house_id = level.house.id


class Node:
    def __init__(self, dict_, level):
        self.__dict__ = dict_
        # if reflection, switch to the mirrored model and adjust transform
        if hasattr(self, 'transform') and hasattr(self, 'modelId'):
            t = np.asarray(self.transform).reshape(4,4)
            if np.linalg.det(t) < 0:
                t_reflec = np.asarray([[-1, 0, 0, 0], \
                                      [0, 1, 0, 0], \
                                      [0, 0, 1, 0], \
                                      [0, 0, 0, 1]])
                t = np.dot(t_reflec, t)
                self.modelId += '_mirror'
                self.transform = list(t.flatten())


class House:
    def __init__(self, house_json, arch_json=None):
        self.__dict__ = json.loads(open(house_json, 'r').read())
        self.levels = [Level(l,self) for l in self.levels]
        self.rooms = [r for l in self.levels for r in l.rooms]
        self.nodes = [n for l in self.levels for n in l.nodes]
        self.node_dict = {id_: n for l in self.levels for id_,n in l.node_dict.items()}

        if arch_json:
            arch = json.loads(open(arch_json, 'r').read())
            self.walls = [w for w in arch['elements'] if w['type'] == 'Wall']
