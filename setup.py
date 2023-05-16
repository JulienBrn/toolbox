from distutils.core import setup


setup(
    name='toolbox',
    packages=['toolbox'],
    version='0.1',
    license='MIT',
    description = 'Toolbox for python analysis',
    description_file = "README.md",
    author="Julien Braine",
    author_email='julienbraine@yahoo.fr',
    url='https://github.com/JulienBrn/toolbox',
    download_url = 'https://github.com/JulienBrn/toolbox.git',
    package_dir={'': 'src'},
    keywords=['python', 'dataframe'],
    install_requires=['pandas', 'matplotlib', 'PyQt5', "sklearn", "scipy", "numpy", "scikit-learn", "mat73", "psutil"],
)