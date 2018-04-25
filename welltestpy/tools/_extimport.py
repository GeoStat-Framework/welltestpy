# -*- coding: utf-8 -*-
"""External Imports for welltestpy.
"""

from __future__ import absolute_import, division, print_function

import sys

# import matplotlib
# use the Qt5 Frontend of matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('WXAgg')
# import matplotlib.pyplot as plt
# suppress the standard keybinding 's' for saving
# plt.rcParams['keymap.save'] = ''
# use the ggplot style like R
# plt.style.use('ggplot')

PY3 = sys.version_info[0] == 3

if PY3:
    import io
    StrIO = io.StringIO
    BytIO = io.BytesIO
else:
    import StringIO
    StrIO = BytIO = StringIO.StringIO
