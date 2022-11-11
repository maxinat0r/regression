CURRENT_VERSION = "0.0.1"
CODE_NAME = "regression"

setup_configuration = {
    "name": CODE_NAME,
    "version": CURRENT_VERSION,
    "description": "Regression",
    "author": "Max de la Vieter",
    "author_email": "max.delavieter@coolblue.nl",
    "include_package_data": True,
    "install_requires": [
        "numpy==1.21.6",
        "pandas==1.3.5",
    ],
    "tests_require": ["pytest"],
    "python_requires": ">=3.7",
}
