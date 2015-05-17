from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


extensions = [
    Extension("ecdsa_recover", ["ecdsa_recover/ecdsa_recover.pyx"],
              libraries=['gmp', 'm'],
              extra_compile_args=['-O3']
              )
]
compiler_directives = {}
install_requires = ['bitcoin']

version = '0.1.2'

setup(
    #ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    name='ecdsa_recover',
    version=version,
    description="faster implementation of ecdsa recover using cython + gmp",
    author="HeikoHeiko",
    author_email='heiko@ethdev.com',
    url='https://github.com/heikoheiko/cy_ecdsa_recover',
    license="BSD",
    keywords='ecdsa recover ethereum cython gmp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=install_requires,
)
