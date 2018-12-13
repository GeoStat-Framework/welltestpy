# -*- coding: utf-8 -*-
"""External Imports for welltestpy.
"""

from __future__ import absolute_import, division, print_function

import sys

PY3 = sys.version_info[0] == 3

if PY3:
    import io

    StrIO = io.StringIO
    BytIO = io.BytesIO
else:
    import StringIO

    StrIO = BytIO = StringIO.StringIO
