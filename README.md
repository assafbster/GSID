# GSID
Sequential Information Distilling
This Python project contains the code used for the experimental results in the "Sequential Quantization for Inference Applications" paper.

The main file is :runGMclassifyGit.py
It contains a nested loop: one on the databases and on the classifiers, according to the following sets:

classifiernames = ["SGM","RF","NB","Logit"]
databasename = ['MNIST','BSCRepetition','BSCDouble','BSCvariable','BSC integrated']
Then, the variables "databaseinds", and "classifiersinds" should be set to select the appropriate classifiers and databases.
Note that "M" of SGM is set in the code, and it can take a different value according to the database.

In addition, the file TestBinaryMulticlassGit.py contains the script that implements the information distilling solution to the full (i.e. ten digit) MNIST corpus.
It does so using a hierarchical tree structure, as explained in the paper.
