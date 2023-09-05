import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='symbal',
    version='0.0.21',
    author='Alex Summers',
    author_email='ajs0201@auburn.edu',
    description='A Python package for batch adaptive sampling with symbolic regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ajsummers/symbal',
    project_urls={
        'Issues': 'https://github.com/ajsummers/symbal/issues'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
)
