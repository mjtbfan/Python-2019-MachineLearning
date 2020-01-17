Nicholas Delli Carpini

The code for this project was tested using Python 3.8 64-bit on Windows 10:
The following libraries were used in the program
-Numpy | 1.17.3+mkl
-Pandas | 0.25.3
-MatPlotLib | 3.2.0rc1
-Mlxtend | 0.17.0
-SciKit-Learn | 0.21.3
-Scipy | 1.3.1
-Xlrd | 1.2.0

Using Windows 10, a number of libraries could not be installed using "pip install %name%" 
for some reason; therefore, the following libraries were downloaded from 
https://www.lfd.uci.edu/~gohlke/pythonlibs/
-numpy+mkl
-pandas
-matplotlib
-sklearn
-scipy
To install these libraries - do "pip install %file%.whl" where file is downloaded from the above site

Each question requiring code is broken up into their own .py files, and simply require
"python %file%.py" to run given the above libraries are installed. Additionally, data files should
not be moved from their folders without properly updating the code references to these files.