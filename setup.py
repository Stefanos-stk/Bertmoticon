import pathlib
from setuptools import setup, find_packages


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bertmoticon",
    version="1.0.0",
    description="multilingual emoji prediction",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Stefanos-stk/Bertmoticon",
    author="Stefanos Stoikos",
    author_email="st.stoikos@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    #added one extra step to the package dir so that the call inside __init__.py is just DATA_PATH + 'model'
    #package_dir ={'bertmoticon': 'bertmoticon/'
    #},
    include_package_data=True,
    install_requires=["torch","transformers","requests"],
    entry_points={
        "console_scripts": [
        ]
    },
)