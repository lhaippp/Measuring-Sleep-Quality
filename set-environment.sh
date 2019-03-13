#!/bin/sh
conda install -c anaconda numpy 
conda install pandas
conda install scikit-learn
pip3 install keras
conda install -c anaconda seaborn 
conda install -c conda-forge tensorflow 
python -m pip install -U matplotlib
conda install -c conda-forge lightgbm 

echo "****************************************************************"
echo "Congratulations! All the environment are installed successfully"
echo "****************************************************************"
echo "Now you enter the project!"
echo "****************************************************************"
