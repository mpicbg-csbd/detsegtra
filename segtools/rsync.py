from subprocess import run, Popen
from pathlib import Path
import shutil
from .ns2dir import load,save


local_push    = Path("/Users/broaddus/Desktop/Projects/devseg_2_local/")
local_pull    = Path("/Users/broaddus/Desktop/project-broaddus/")
remote        = Path("/projects/project-broaddus/")


def rsync_pull2(targetfile="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/", cleardir=False, justfiles=False, return_value=True, excludes=[]):
  targetfile = Path(targetfile)
  shared_extension = str(targetfile).replace("/lustre/","").replace("/projects/project-broaddus/","").replace("/Users/broaddus/Desktop/project-broaddus/","")
  # extension = 
  # localpath = localpath.replace("/lustre/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  # localpath = localpath.replace("/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  # localpath = Path(localpath)
  # shared_extension = str(localpath.relative_to(local_pull))  
  # import ipdb; ipdb.set_trace()
  if not targetfile.suffix:
    if shared_extension[-1] != '/': shared_extension += '/'
    localdir = (local_pull / shared_extension).parent
  else:
    localdir = (local_pull / shared_extension).parent

  if cleardir:
    if localdir.exists(): shutil.rmtree(localdir)

  args = ""
  args += " --exclude '*.pt' --exclude 'pred/' "
  _s = " --exclude '{}' "*len(excludes)
  args += _s.format(*excludes)

  # args += " --include 'pred/mx_z/' "
  # args += " --exclude '*vali*' "
  if justfiles: args += " --exclude '*/'"

  ## rsync
  # user@<source>:<source_dir> <dest_dir>
  flags = ' -mapHAXxv --numeric-ids --delete --progress -e "ssh -T -c arcfour -o Compression=no -x" '
  flags = ' -mapv --numeric-ids --delete --progress '
  args = flags + args
  run([f"mkdir -p {localdir}"],shell=True)
  cmd = f"rsync {args} efal:{remote}/{shared_extension} {local_pull}/{shared_extension} > rsync.out 2>&1"
  
  # x = {k:v for k,v in locals().items() if k in ['localdir', 'cmd', 'targetfile', 'shared_extension']}
  # import json
  print(cmd)
  # return

  if return_value:
    run([cmd], shell=True)
    x=local_pull/shared_extension
    # print(json.dumps(locals(),sort_keys=True, indent=2, default=str))
    return load(x)
  else:
    Popen([cmd], shell=True)



def rsync_pull(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/", cleardir=False, justfiles=False, return_value=True, _pull=True):
  
  if _pull:
    localpath = localpath.replace("/lustre/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
    localpath = localpath.replace("/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  else:
    localpath = localpath.replace("/lustre/projects/project-broaddus/","/Users/broaddus/Desktop/project-broaddus/")
  localpath = Path(localpath)
  shared_extension = str(localpath.relative_to(local_pull))
  
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1] != '/': shared_extension += '/'

  if cleardir:
    if localpath.exists(): shutil.rmtree(localpath)

  args = ""
  args += " --exclude '*.pt' --exclude 'pred/' "
  # args += " --include 'pred/mx_z/' "
  # args += " --exclude '*vali*' "
  if justfiles: args += " --exclude '*/'"

  ## rsync
  # user@<source>:<source_dir> <dest_dir>
  flags = ' -mapHAXxv --numeric-ids --delete --progress -e "ssh -T -c arcfour -o Compression=no -x" '
  flags = ' -mapv --numeric-ids --delete --progress '
  args  = flags + args

  if _pull:
    run([f"mkdir -p {localpath}"],shell=True)
    cmd = f"rsync {args} efal:{remote}/{shared_extension} {local_pull}/{shared_extension} > rsync.out 2>&1"
  else:
    cmd = f"rsync {args} {local_pull}/{shared_extension} efal:{remote}/{shared_extension} > rsync.out 2>&1"
    return_value=False

  print(cmd)
  print(localpath)

  if return_value:
    run([cmd], shell=True)
    return load(localpath)
  else:
    Popen([cmd], shell=True)




# def qsave(obj, dir='./'):
#     save(obj, dir + "qsave")

# def qload(): return rsync_pull("/Users/broaddus/Desktop/project-broaddus/devseg_2/src/qsave.tif")

def rsync_push(localpath="/Users/broaddus/Desktop/project-broaddus/devseg_2/e02/test/"):

  localpath  = Path(localpath)
  local_base = Path("/Users/broaddus/Dropbox/Projects_Shared/")
  local_base = Path("/Users/broaddus/Desktop/push-broaddus/")
  shared_extension = str(localpath.relative_to(local_base))
  if localpath.suffix: localpath = localpath.parent
  elif shared_extension[-1]!='/': shared_extension += '/'

  cmd = f"rsync -maP {local_base}/{shared_extension} efal:{remote}/{shared_extension} > rsync.out 2>&1"
  Popen([cmd],shell=True)