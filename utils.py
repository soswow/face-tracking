import os
import random

def directory_files(path):
    for _, _, files in os.walk(path, topdown=False):
        print "Yielding %d files ... " % len(files)
        for name in files:
            yield os.path.join(path, name), name

def random_from_generator(gen, every_n, until=None):
    next = random.randint(0, every_n)
    a = 0
    for i, k in enumerate(gen):
        if i == a+next:
            yield k
            a, next = i, random.randint(1,every_n) + ((every_n-next) if every_n > next else 0)
        if until and i >= until:
            return