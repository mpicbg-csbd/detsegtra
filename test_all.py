

def test_scikit_image_regionprops():
  def f(a):
    try:
      val = getattr(rp[0],a)
    except (NotImplementedError, AttributeError) as e:
      print(e)
      val = None
    # print(a)
    return val
  return [f(a) for a in dir(rp[0])]

