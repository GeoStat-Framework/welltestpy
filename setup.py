#!/usr/bin/env python
"""
wtp: WellTestPy

welltestpy is a general package offering data-classes to handle well-based Field-campaigns.

"""
DOCLINES = __doc__.split("\n")

readme = open('README').read()

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Natural Language :: English
Operating System :: MacOS
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Software Development
Topic :: Utilities
"""

MAJOR               = 0
MINOR               = 2
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

from setuptools import setup, find_packages

metadata = dict(
    name = 'welltestpy',
    version=VERSION,
    maintainer = "Sebastian Mueller",
    maintainer_email = "sebastian.mueller (at) ufz (dot) de",
    description = DOCLINES[0],
    long_description = readme,
    author = "Sebastian Mueller",
    author_email = "sebastian.mueller (at) ufz (dot) de",
    license = 'LGPL -  see LICENSE',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
#    install_requires=['numpy', 'scipy'],
    packages=find_packages(),
    )

setup(**metadata)
