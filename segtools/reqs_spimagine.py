from . import lib as ll

import spimagine

def rethresh_nuc(nuc, img, hyp, pad, newthresh=0.75):
  img_crop = ll.nuc2img(nuc, img, pad)
  hyp_crop = ll.nuc2img(nuc, hyp, pad)
  lab, ncells = nd.label(img_crop > newthresh)
  spimagine.volshow([img_crop, hyp_crop, lab], interpolation='nearest')
  input('quit?')
