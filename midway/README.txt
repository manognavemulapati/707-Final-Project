benchmark.py can be run with 2 system arguments like so:
python benchmark.py sampto datanum

Where sampto is how many samples to take in from the data, with maximum 650000.
datanum is the number corresponding to a dataset, which can be found in the data folder.
Examples are 100, 101, 102, ... These are datasets taken from the MIT-BIH database:
https://www.physionet.org/physiobank/database/mitdb/

It prints the number of peaks found in the data, the number of times
it was correct in classifying them, and the total accuracy.
It also outputs tuples (start, end) of what labels were found where throughout the file,
and it stores these in the file output.txt

In the interest of space only data 100 (so datanum=100) is stored in the data folder, and the model folder should have the trained model found here:
https://drive.google.com/file/d/1aFKVKz41A9fu8dX2KfwlEGV8vz9ljiuZ/view