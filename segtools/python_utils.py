from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
# from functools import reduce
from sortedcollections import SortedDict
import collections
from collections import Counter
from tabulate import tabulate
from math import floor, ceil
from pathlib import Path
from glob import glob
import re


flatten = lambda l: [item for sublist in l for item in sublist]

def factors(n):
  "from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python"
  return set(reduce(list.__add__,
    ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))

def pfactors(n): 
  """
  Finds the prime factors of 'n'
  https://stackoverflow.com/questions/14550794/python-integer-factorization-into-primes
  """ 
  from math import sqrt 
  pFact, limit, check, num = [], int(sqrt(n)) + 1, 2, n 
  if n == 1: return [1] 
  for check in range(2, limit): 
       while num % check == 0: 
          pFact.append(check) 
          num /= check 
  if num > 1:
    pFact.append(num)
  return pFact

def rowscols(n,cols=8):
  "divide n things up into rows*columns things"
  rows,xt = divmod(n,cols)
  if rows == 0:
    rows,cols = 1,xt
  return rows, cols

def timewindow(lst, t, l):
  "window of fixed length l into list lst. try to center around t."
  assert l <= len(lst)
  if t < l//2: t=l//2
  if t >= len(lst) - l//2: t=len(lst) - ceil(l/2)
  return lst[t-l//2:t+ceil(l/2)]

def print_sorted_counter(l):
  s = SortedDict(Counter(l))
  print(tabulate([s.keys(), s.values()]))

def sorted_alphanum( l ):
  """ Sort the given iterable in the way that humans expect.
      taken from https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  return sorted(l, key = alphanum_key)

def path_base_ext(fname):
  directory, base = os.path.split(fname)
  base, ext = os.path.splitext(base)
  return directory, base, ext

def pprint_list_of_dict(lod, keys=None):
  if not keys:
    keys = lod[0].keys()
  table = [keys]
  table = table + [d.values() for d in lod]
  print(tabulate(table))

def timing(f):
  @wraps(f)
  def wrap(*args, **kw):
    ts = time()
    result = f(*args, **kw)
    te = time()
    print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
    return result
  return wrap

def do(iter):
  return [x for x in iter]

def groupbyn(list0, n):
  return [list0[i:i+n] for i in range(0, len(list0), n)]

## uwe's stuff

def _raise(e):
  raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
  collections.deque(iterator, maxlen=0)

def compose(*funcs):
  return lambda x: reduce(lambda f,g: g(f), funcs, x)

def pipeline(*steps):
  return reduce(lambda f,g: g(f), steps)


## print type hierarchy for arbitrary objects

# def printtypes(obj):
#   if hasattr(obj, '__len__'):
#     print(type(obj))
#     for x in obj:
#       printtypes(x)

def parse_python_script_comments(filename):
  lines = open(filename,'r').readlines()
  block_indices = [i for i,line in enumerate(lines) if '"""' in line]
  block_indices = groupbyn(block_indices,2)
  textlist = []
  for bi in block_indices:
    varname = lines[bi[0]][:-6] # remove last bit from eg (info = """)
    vartext = ''.join(lines[bi[0]+1:bi[1]])
    textlist.append([varname, vartext])
  return textlist

def glob_and_parse_filename(globname, n=3):
  try:
    lastfile = Path(sorted(glob(globname))[-1])
    lastfile_number = int(lastfile.stem[-n:])
    return lastfile_number
  except IndexError as e:
    print(e)
    return None