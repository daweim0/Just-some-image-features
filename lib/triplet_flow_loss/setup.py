from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

extensions = [Extension("*", ["*.pyx"])]

setup(
    ext_modules = cythonize(extensions)
)
