# early-vision-toolbox
a collection of models and analysis methods for early visual areas.

## TODOs


## about SPAMS

1. the SPAMS here is from <https://github.com/samuelstjean/spams-python>, which seems to be newer than on the one the official site, in terms of SWIG code for binding.
2. seems that I can make OpenBLAS work with it, by changing the `setup.py`. The following works on Mac OS X 10.10, with OpenBLAS (you must install `gfortran` first to make LAPACK also compiled) installed, and Numpy 1.11.0.

~~~python
import os
#os.environ['DISTUTILS_DEBUG'] = "1"
from distutils.core import setup, Extension
import distutils.util
import numpy

# includes numpy : package numpy.distutils , numpy.get_include()
# python setup.py build --inplace
# python setup.py install --prefix=dist, 
incs = ['.'] + map(lambda x: os.path.join('spams',x),[ 'linalg', 'prox', 'decomp', 'dictLearn']) + [numpy.get_include()] + ['/Users/yimengzh/miniconda/envs/early-vision-toolbox/include/python2.7']

osname = distutils.util.get_platform()
cc_flags = ['-fPIC', '-I/opt/OpenBLAS/include']
link_flags = ['-s', '-L/opt/OpenBLAS/lib']
libs = ['stdc++', 'openblas']
libdirs = []

# if osname.startswith("macosx"):
#     cc_flags = ['-fPIC', '-fopenmp','-m32']
#     link_flags = ['-m32', '-framework', 'Python']
~~~

(later part doesn't need changing)

Notice that `if osname.startswith("macosx")` part can be ignored. Simply

~~~bash
python install.py build
python install.py install
~~~

would install the package, as long as your current Python environment is the one you want to use the whole `early-vision-toolbox`.

Another interesting thing is that `import spams` will fail if not `import numpy` first... Don't know if this only applies to Mac OS X, though.

~~~
>>> import spams
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/yimengzh/miniconda/envs/early-vision-toolbox/lib/python2.7/site-packages/spams.py", line 6, in <module>
    import spams_wrap
  File "/Users/yimengzh/miniconda/envs/early-vision-toolbox/lib/python2.7/site-packages/spams_wrap.py", line 32, in <module>
    _spams_wrap = swig_import_helper()
  File "/Users/yimengzh/miniconda/envs/early-vision-toolbox/lib/python2.7/site-packages/spams_wrap.py", line 28, in swig_import_helper
    _mod = imp.load_module('_spams_wrap', fp, pathname, description)
ImportError: dlopen(/Users/yimengzh/miniconda/envs/early-vision-toolbox/lib/python2.7/site-packages/_spams_wrap.so, 2): Symbol not found: _dgesvd_
  Referenced from: /Users/yimengzh/miniconda/envs/early-vision-toolbox/lib/python2.7/site-packages/_spams_wrap.so
  Expected in: dynamic lookup
~~~

importing `numpy` first works.

~~~
>>> import numpy
>>> import spams
>>>
~~~