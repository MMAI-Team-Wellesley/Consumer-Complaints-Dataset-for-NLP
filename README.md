# Consumer-Complaints-Dataset-for-NLP

Since we will be working on this project in 3 teams, modifying the same file at the same time might cause merge
conflict. 
~~For now, to avoid possible merge conflicts, I've devided the project into 2 files: preprocessing.py/ipynb and~~
~~model.py/ipynb.~~

<strong><em>I wrote the code framework as .py files and converted them into .ipynb. However if you modify the .py file, the corresponding .ipynb file won't be changed automatically and vice versa.</em></strong> That means we will need to decide whether to only work on the .py files or to only work on the .ipynb files, otherwise there might be inconsistency.

Team 2(Norman, Kathleen) will modify the preprocessing.py file to create functions which can perform data cleaning and pre-processing. Importing the dataset is not necessary in this file, but feel free to import it and play around and see the result of pre-processing.

Team 1(Gen, Kishawna) and Team 3(Ankan, Drake) will be working on the models. ~~the model.py/ipynb file to build model and do~~
~~the validation. Feel free to let me know if you would like to separate it into 2 different parts to reduce conflicts. The preprocess_data function has already been imported from preprocessing.py, so you only need to import the dataset and call the preprocess_data function on it.~~

<strong>Please make sure the master branch never fails to compile, don't merge any error code into master branch. Usually we don't want to directly merge our code into master branch since there might be merge conflicts and might crash the master branch. But to simplify the process, you can merge your code into master once you have confirmed that the code is building with no errors, let me know if any conflicts happened and I'll try to help. And it is a best practice to frequently check for updates in remote master branch and merge it into your local code.</strong>

I might write a documentation for setting up git on your computer later.

~~We can modify this README file and display our outcomes after we've finished the project.~~

I'm happy to help if you have any questions!

# Essential files:
1. complaints_processed.csv
2. customer_complaints_clean_vF.csv, please note that the dataset on github might not be complete because of the file size limit, if that's the case please run the preprocessing code to generate the cleaned dataset.
3. For the v2 dataset, you need to down load some pre-trained GloVe word embedding file (in our case glove.6B.100d.txt) from https://nlp.stanford.edu/projects/glove/ .
4. For models, if you are going to load dataset locally or on google colab, you might need to change the dataset file path. If you are using google colab, please make sure you upload all the necessary files.
