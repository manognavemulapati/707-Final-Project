First, download the data from the MIT-BIH database (.atr, .dat, and .hea files) and put them in the "data" folder. More info about the data here: 
 https://www.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm
The data can also be found at our github:
 https://github.com/manognavemulapati/707-Final-Project

Then run signalparse.py and imageparse.py, which will take in data from the "data" folder and eventually output it into "dataparse".

Then combinedata.py will collect all the data in "dataparse", randomly shuffle it, and output it into "finaldata".

Now you can run dimreduction.py with one system argument from [ISOMAP, LLE, Spectral] ex:
  python dimreduction.py ISOMAP
which will take in data from "finaldata" and apply the specified dimensionality reduction 
technique, outputting the reduced data into the corresponding folder of the same name.

Lastly, to train the model use train.py with one system argument from [ISOMAP, LLE, Spectral, Normal] ex:
  python train.py Normal
which will take in data from the corresponding folder and train the model 
(data from "finaldata" is used with argument "Normal")

Finally, plotting.py will plot the training and validation accuracies under each set of data,
 as well as some examples of ISOMAP dimensionality reductions for different heartbeats.