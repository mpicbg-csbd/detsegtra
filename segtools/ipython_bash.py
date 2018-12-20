from subprocess import check_output, run
from pathlib import Path
from termcolor import colored
from glob import glob
import matplotlib.pyplot as plt



def edit(num=None):
  if num is None:
    cmd = 'rsub --port 52699 train_and_seg.py'
  else:
    cmd = 'rsub --port 52699 training/t{:03d}/train_and_seg.py'.format(num)
  return run([cmd], shell=True)

def errs(globstring, n=4):
  # !grep -C $n "Error" training/t04?/stderr.txt
  cmd = 'grep --color=always -C {0} "Error" training/{1}/stderr.txt'.format(n, globstring)
  res = check_output([cmd], shell=True, universal_newlines=True)
  # res = res.replace('AssertionError', colored('AssertionError', 'red'))
  print(res)

def rsub(name):
  cmd = 'rsub --port 52699 {}'.format(name)
  return run([cmd], shell=True)

def job(n_load, n_save):
  cmd = './job_starter.py train_and_seg.py training/t{:03d} training/t{:03d}'.format(n_load, n_save)
  return run([cmd], shell=True)

def diff(globstring, filename='train_and_seg.py'):
  files = Path('training/') / globstring / filename
  files = glob(str(files))
  assert len(files) >= 2
  difflist = []
  for i in range(len(files)-1):
    f0 = files[i]
    f1 = files[i+1]
    cmd = 'diff {0} {1}'.format(f0, f1)
    print(cmd)
    res = run([cmd], shell=True, universal_newlines=True)
    print(res)
    print()
    difflist.append(res)
  return difflist

def compare_histories(globstring):
  files = Path('training/') / globstring / 'history.txt'
  files = sorted(glob(str(files)))
  assert len(files) >= 2
  histories = []
  plt.figure()
  for f0 in files:
    f0 = Path(f0)
    his = eval(f0.open('r').read())
    histories.append(his)
    plt.plot(his['loss'], label='loss ' + f0.parts[-2])
    plt.plot(his['val_loss'], label='val_loss ' + f0.parts[-2])
  plt.legend()
  return histories

def compare_segscores(globstring):
  files = Path('training/') / globstring / 'SEG.txt'
  files = sorted(glob(str(files)))
  for f0 in files:
    with open(f0, 'r') as fin:
      print(f0)
      print(fin.read())

def example_colored_output():
  print(colored('hello', 'red'), colored('world', 'green'))
  print(colored("hello red world", 'red'))
