# Machine Comprehension using Commonsense Knowledge
## SemEval-2018 Task 11

Neural network with theano: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano

There is anaconda environment exported in the file environment.yml. To import it use:

conda env create -f myenv.yml

There are instructions for downloading the glove vectors (word embedings) and bAbi data (question answering corpus).

Thre train and test files are hardcoded in utils.py

To train the neural network call:
    ./main.py --network dmn_qa --babi_id 1

To test the neural network call:
    ./main.py --network dmn_qa --mode test --load_state *.state file* 
where *.state file* is file with the trained weights located into the states folder.
I recomend dmn_qa.mh5.n40.bs10.babi1.epoch6.test1.22623acc64.23077.state where acc64.230077 is the accuracy (64.230077%)

This will create output.txt file with the wrong answered questions and those question that the network is uncertain about.


