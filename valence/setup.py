"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='elderly-emotion-sc',  # Required
    version='0.0.0',  # Required
    description='Entry for the Elderly Emotion Subchallenge Interspeech 2020',  # Optional

    url='https://github.com/gizemsogancioglu/elderly-emotion-SC',  # Optional

    author='Gizem Sogancioglu',

    author_email='gizemsogancioglu@gmail.com',

    classifiers=[  # Optional
        'Programming Language :: Python :: 3.7',
    ],

    keywords='machine learning, emotion classifier, development',

    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'scripts'},  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().

    packages=find_packages(where='scripts'),  # Required

    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.7, <4',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    dependency_links=[
                       'git+https://github.com/flairNLP/flair.git#egg=flair'
                   ],
    install_requires=['pandas>= 1.1.1 ',
                      'scikit-learn==0.22.2.post1',
                      'joblib==0.16.0',
                      'fasttext>= 0.9.2',
                      'textblob>= 0.15.3',
                      'flair',
                      'xlrd==1.2.0'
                      ],  # Optional
    zip_safe = True,

    entry_points={
        'console_scripts': [
            'valence=valence_classifier:main',
        ],
    },

    project_urls={
        'Bug Reports': 'https://github.com/gizemsogancioglu/elderly-emotion-SC/issues',
        'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': 'https://github.com/gizemsogancioglu/elderly-emotion-SC',
    },
)