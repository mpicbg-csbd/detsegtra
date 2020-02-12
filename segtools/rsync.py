from subprocess import call, Popen
from pathlib import Path


local_push    = Path("/Users/broaddus/Desktop/Projects/devseg_2_local/")
local_pull    = Path("/Users/broaddus/Desktop/project-broaddus/")
remote        = Path("/projects/project-broaddus/")

def rsync_pull(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/",cleardir=False,justfiles=False):

  localpath = localpath.replace("/lustre/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  localpath = localpath.replace("/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")

  localpath = Path(localpath)
  shared_extension = str(localpath.relative_to(local_pull))
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1]!='/': shared_extension += '/'
  
  if cleardir:
    if localpath.exists(): shutil.rmtree(localpath)

  excludes = "--exclude '*.pt' --exclude 'pred/'"
  excludes += " --exclude ta/mx_vali/target.tif "
  if justfiles: excludes += " --exclude '*/'"

  call([f"mkdir -p {localpath}"],shell=True)
  cmd = f"rsync -maP {excludes} efal:{remote}/{shared_extension} {local_pull}/{shared_extension} > rsync.out 2>&1"
  print(cmd)
  Popen([cmd],shell=True)

def rsync_push(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/"):

  localpath  = Path(localpath)
  local_base = Path("/Users/broaddus/Dropbox/Projects_Shared/")
  local_base = Path("/Users/broaddus/Desktop/push-broaddus/")
  shared_extension = str(localpath.relative_to(local_base))
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1]!='/': shared_extension += '/'

  cmd = f"rsync -maP {local_base}/{shared_extension} efal:{remote}/{shared_extension} > rsync.out 2>&1"
  Popen([cmd],shell=True)