Data Analysis in Python: pandas in practice
===========================================

**Author:** Jan Smitka, <jan.smitka@lynt.cz>

This repository contains code examples and data for a talk about data analysis in Python, which was presented
at Pilsen Pyvo community event on 9th May, 2018.

Main goal of the talk is to demonstrate basics of the library on a real-world example — performance evaluation
of PPC campaigns in AdWords. This will include data loading, computations, filtering, sorting, grouping and joining.


Structure of the repository
---------------------------

The repository contains the following folders:

* `data`: pre-generated data sets that were used for demonstration during the talk. Please note that the data is not
based on any real data of our clients. Names for the campaigns and ad groups are based on category names
at [e-sportshop.cz](https://e-sportshop.cz/) with kind permission from RADANSPORT s.r.o.
* `generator`: Tool that was used to generate the data.
* `notebooks`: iPython notebooks that were shown during the talk, including comments.
* `slides`: slides of the talk, with all related sources, such as images, source code snippets and so on.


Dependencies
------------

The examples requires Python 3.5 or higher. Python 3.6 is recommended and required to run the data generator.

In order to run the examples, you'll need to install dependencies from the requirements.txt file:

    pip install -r requirements.txt

If you would like to install the packages manually, install at least the following packages:

    pip install pandas numpy bottleneck xlrd xlsxwriter SQLAlchemy mysqlclient
    

Generating the data
-------------------

If you would like to re-generate the sample data, just run the following command:

```
python generator/generator.py
```


Running the notebook
--------------------

Jupyter Lab is recommended for running the notebook:

```
jupyter lab
```

Then open `notebooks/ppc-examples.ipynb` from the interface.


License
-------

This work and all of its parts is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

This means that you are free to: 

* **Share** — copy and redistribute the material in any medium or format.
* **Adapt** — remix, transform, and build upon the material for any purpose, even commercially. 

You can read the whole license in the LICENSE.md file.
