from subprocess import run, Popen
from pathlib import Path
import shutil
from .ns2dir import load,save


local_push    = Path("/Users/broaddus/Desktop/Projects/devseg_2_local/")
local_pull    = Path("/Users/broaddus/Desktop/project-broaddus/")
remote        = Path("/projects/project-broaddus/")

def rsync_pull(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/",cleardir=False,justfiles=False,async=True):

  localpath = localpath.replace("/lustre/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  localpath = localpath.replace("/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")

  localpath = Path(localpath)
  shared_extension = str(localpath.relative_to(local_pull))
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1]!='/': shared_extension += '/'
  
  if cleardir:
    if localpath.exists(): shutil.rmtree(localpath)

  args = ""
  args += " --exclude '*.pt' --exclude 'pred/' "
  # args += " --include 'pred/mx_z/' "
  # args += " --exclude '*vali*' "
  if justfiles: args += " --exclude '*/'"

  ## rsync 
  flags = ' -mapHAXxv --numeric-ids --delete --progress -e "ssh -T -c arcfour -o Compression=no -x" ' #user@<source>:<source_dir> <dest_dir>
  flags = ' -mapv --numeric-ids --delete --progress '
  args = flags + args
  run([f"mkdir -p {localpath}"],shell=True)
  cmd = f"rsync {args} efal:{remote}/{shared_extension} {local_pull}/{shared_extension} > rsync.out 2>&1"
  print(cmd)
  _call = Popen if async else run
  _call([cmd],shell=True)

def qload():
    rsync_pull("/Users/broaddus/Desktop/project-broaddus/devseg_2/src/qsave.tif", async=False)
    img = load("/Users/broaddus/Desktop/project-broaddus/devseg_2/src/qsave.tif")
    return img

# def qsave(obj, dir='./'):
#     save(obj, dir + "qsave")

def rsync_push(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/"):

  localpath  = Path(localpath)
  local_base = Path("/Users/broaddus/Dropbox/Projects_Shared/")
  local_base = Path("/Users/broaddus/Desktop/push-broaddus/")
  shared_extension = str(localpath.relative_to(local_base))
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1]!='/': shared_extension += '/'

  cmd = f"rsync -maP {local_base}/{shared_extension} efal:{remote}/{shared_extension} > rsync.out 2>&1"
  Popen([cmd],shell=True)