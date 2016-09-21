
import os
import os.path



def isFileMoreRecentThan(a,b):
    """
    think of it as: return true if a needs to be rebuilt having dependencies b
    b can be a list of files or a single file
    returns true if
    - a doesn't exist
    - a is more recent than (at least one) b
    - (at least one) b doesn't exist
    returns false if
    - a is older than (all) b

    """
    if type(b) != list:
        b = [b]
    if os.path.exists(a):
        at = os.path.getmtime(a)
        print [not os.path.exists(i) for i in b]
        if any([not os.path.exists(i) for i in b]):
            return True
        if any([os.path.getmtime(i) < at for i in b]):
            return True
        return False
    return True

def ls(p, where=lambda x: True):
    return [os.path.join(p, i) for i in os.listdir(p) if where(os.path.join(p,i))]

def lsdirs(p, where=lambda x: True):
    paths = ls(p)
    return [i for i in paths if (where(i) and os.path.isdir(i))]
