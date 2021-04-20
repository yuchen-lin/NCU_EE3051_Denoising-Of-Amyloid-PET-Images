import os

def mfdr(paths):
    for path in paths:
        try:
            os.mkdir(path)
        except:
            pass