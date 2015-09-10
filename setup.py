from setuptools import setup, Extension


extensions = [
    Extension("ecdsa_recover", ["ecdsa_recover/ecdsa_recover.pyx"],
              libraries=['gmp', 'm'],
              extra_compile_args=['-O3', '-I/usr/include', '-I/usr/local/include']
              )
]
compiler_directives = {}
install_requires = ['bitcoin']
setup_requires = ['cython']

version = '0.1.6'


setup(
    #ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    ext_modules=extensions,
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
    setup_requires=setup_requires,
)
