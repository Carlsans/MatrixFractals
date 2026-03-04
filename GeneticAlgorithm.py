import random
from operator import itemgetter

import numpy as np

from src.Configuration import Configuration


class GeneticAlgorithm:
    def __init__(self):
        self.conf = Configuration()
        self.pool = []
        self.note_sum = 0
    def add_item(self,item,note):
        self.pool.append({"fractal":item,"note":note})
        self.pool = sorted(self.pool, key=itemgetter('note'),reverse=True)
    def pick_item(self):
        return random.choices(self.pool, weights=[item['note'] for item in self.pool], k=1)[0]
    def pick_top_item(self):
        return self.pool[0]
    def kill_overpooled_items(self):
        count = len(self.pool)-self.conf.poolsize
        if count < 1 :
            return
        self.pool = sorted(self.pool, key=itemgetter('note'),reverse=True)
        self.pool = self.pool[0:self.conf.poolsize]

    def scale_notes(self,scalor):
        for item in self.pool:
            item['note'] = item['note']*scalor

    def __str__(self):
        return_string = ""
        for i in self.pool:
            return_string += "fractal : "+ str(i['fractal']['fractal_name']) + " note : "+str(i['note']) +"\n"
        return  return_string
ga = GeneticAlgorithm()
ga.add_item({'fractal_name':"blou1", 'matrix': "matrix1"},.1)
ga.add_item({'fractal_name':"blou2", 'matrix': "matrix2"},.2)
ga.add_item({'fractal_name':"blou3", 'matrix': "matrix3"},.4)
ga.add_item({'fractal_name':"blou4", 'matrix': "matrix4"},.8)
ga.add_item({'fractal_name':"blou5", 'matrix': "matrix5"},1.6)
item = ga.pick_item()
print(item)
ga.kill_overpooled_items()
print(ga)