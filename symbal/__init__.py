# Copyright 2023 Alex Summers
# See LICENSE file for more information

import sys

if sys.version_info[0] == 2:
    raise ImportError('SymBAL requires Python 3.7. You have Python 2 installed.')

__all__ = ['TestFunction']
__version__ = '0.0.2'

from symbal.test_function import TestFunction