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


import inspect
import difflib
import collections
from collections import Sequence

try:
  from colorama import Fore, Back, Style, init
  init()
except ImportError:  # fallback so that the imported classes always exist
  class ColorFallback():
    __getattr__ = lambda self, name: ''  
  Fore = Back = Style = ColorFallback()

def diff_func_source(f1,f2):
  """
  works with functions, classes or entire modules
  prints colored diff output to terminal
  makes it easier to put large, similar functions into same file/module
  """
  def color_diff(diff):
    for line in diff:
      if line.startswith('+'):
        yield Fore.GREEN + line + Fore.RESET
      elif line.startswith('-'):
        yield Fore.RED + line + Fore.RESET
      elif line.startswith('^'):
        yield Fore.BLUE + line + Fore.RESET
      else:
        yield line

  lines1 = inspect.getsourcelines(f1)
  lines2 = inspect.getsourcelines(f2)
  diff = color_diff([line for line in difflib.ndiff(lines1[0],lines2[0])])
  for l in diff: print(l,end='')

def flatten(l):
  for el in l:
    if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el

def recursive_map(func, seq):
  def loop(func,seq):
    if isinstance(seq, (list,set,tuple)):
      for item in seq:
        yield type(item)(loop(func,item))
    elif isinstance(seq, dict):
      for k,v in seq.items():
        yield type(v)(loop(func,v))
    else:
      yield func(item)
  return type(seq)(loop(func,seq))

from collections import Collection, Mapping

def recursive_map2(func, data):
  apply = lambda x: recursive_map2(func, x)
  if isinstance(data, Mapping):
      return type(data)({k: apply(v) for k, v in data.items()})
  elif isinstance(data, Collection) and not isinstance(data, str):
      return type(data)(apply(v) for v in data)
  else:
      return func(data)

# sigds are the significance digits
# inputs are lists of names, values and uncertainties respectively
def _print_fres(names, vals, uncs, sigds = 2, rfmt = 'pm', ws = False):
    try:
        if all([str(u).lower() not in 'inf' for u in uncs]):
                sigs = [
                    (re.search('[1-9]', str(u)).start()-2 \
                        if re.match('0\.', str(u)) \
                    else -re.search('\.', str(float(u))).start())+sigds \
                    for u in uncs
                    ]
                # significant digits rule in uncertainty
        else:
            print('Warning: infinity in uncertainty values')
            sigs = [sigds] * len(uncs)
    except TypeError: #NaN or None
        raise TypeError('Error: odd uncertainty values')

    rfmt = rfmt.lower()
    # this can be done better/prettier I think
    if rfmt in ['fancy', 'pms']: # pms stands for pmsign
        res_str = '{{0}} = {{1:{ws}{nfmt}}} ± {{2:{ws}{nfmt}}}'
    elif rfmt in ['basic', 'pm', 'ascii']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}}+/-{{2:{ws}{nfmt}}}'
    elif rfmt in ['tex', 'latex']:
        res_str = '${{0}} = {{1:{ws}{nfmt}}} \\pm {{2:{ws}{nfmt}}}$'
    elif rfmt in ['s1', 'short1']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}} ± {{2:{ws}{nfmt}}}'
        # not yet supported. to do: shorthand notation
    elif rfmt in ['s2', 'short2']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}}({{2:{ws}{nfmt}}})'
    else:
        raise KeyError('rfmt value is invalid')

    for i in range(len(vals)):
        try:
            print((res_str.format(
                    nfmt = '1e' if uncs[i] >= 1000 or uncs[i] <= 0.001 \
                    # 1 decimal exponent notation for big/small numbers
                        else (
                            'd' if sigs[i] <= 0 \
                            # integer if uncertainty >= 10
                            else '.{}f'.format(sigs[i])),
                    ws = ' ' if ws in [True, ' '] else ''
                    )
                 ).format(
                    names[i],
                    round(vals[i], sigs[i]),
                    round(uncs[i], sigs[i])
                    # round to allow non-decimal significances
                 )
             )

        except (TypeError, ValueError, OverflowError) as e:
            print('{} value is invalid'.format(uncs[i]))
            print(e)
            continue
    # to do: a repr method to get numbers well represented
    # instead of this whole mess