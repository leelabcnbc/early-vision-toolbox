from __future__ import print_function, division, absolute_import
import os.path

root_package_spec = __name__
root_path = os.path.normpath(os.path.abspath(os.path.join(os.path.split(__file__)[0])))
print('root dir at {}'.format(root_path))
