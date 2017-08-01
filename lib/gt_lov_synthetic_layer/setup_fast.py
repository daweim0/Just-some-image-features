from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=['-g'],
        extra_link_args=[''],
    )
]

extensions = [Extension("*", ["*.pyx"])]

setup(
    ext_modules = cythonize(extensions)
)
