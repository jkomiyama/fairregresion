#a script for downloading datasets
cd dataset/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data
wget https://github.com/propublica/compas-analysis/archive/master.zip
unzip master.zip
mv compas-analysis-master compas-analysis
