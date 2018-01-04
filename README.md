# Machine Comprehension using Commonsense Knowledge
## SemEval-2018 Task 11

Neural network with theano: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano

There is anaconda environment exported in the file environment.yml. To import it use:

conda env create -f myenv.yml

There are instructions for downloading the glove vectors (word embedings) and bAbi data (question answering corpus).

Thre train and test files are hardcoded in utils.py

To train the neural network call /main.py --network dmn_qa --babi_id 1

