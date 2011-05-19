import os
import random
from os.path import join

def yield_files_in_path(path):
    for top, dirs, files in os.walk(path, topdown=False):
#        print "Yielding %d files ... " % len(files)
        dirs.sort()
        files.sort()
        for name in files:
            yield  join(top, name), name
#        for dir in dirs:
#            print "Accessing %s dir" % dir
#            for file in yield_files_in_path(join(top, dir)):
#                yield file

def directory_files(path):
    output = []
    for top, dirs, files in os.walk(path, topdown=False):
        dirs.sort()
        files.sort()
#        print "Yielding %d files ... " % len(files)
        for name in files:
            output.append((join(top, name), name))
#        for dir in dirs:
#            output += join(top, dir)
    return output

def random_from_generator(gen, every_n, until=None):
    next = random.randint(0, every_n)
    a = 0
    for i, k in enumerate(gen):
        if i == a+next:
            yield k
            a, next = i, random.randint(1,every_n) + ((every_n-next) if every_n > next else 0)
        if until and i >= until:
            return