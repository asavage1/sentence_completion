Improving Sentences using Online Policy Gradient Methods

Bhushan Suwal, Andrew Savage
COMP 150 Reinforcement Learning, Fall 2018

Final Project for Tufts comp 150, a RNN to complete sentences, learning with
reinforcement techniques.

Instructions:

1. In a new Python 3 virtual environment,
        pip install -r requirements.txt

2. cd sentence_completion/word_language_model/

3.  * Train to produce short sentences:
        python3 reinforce.py --num_episodes 5 --num_iter 30 --short_sent

    You should be able to observe the sentences getting shorter. At the end, the directory should have a short_lengths.jpg with the graph of lengths over time, and a short_sent_30.pt as the saved model.

    * Train to produce long sentences:
        python3 reinforce.py --num_episodes 5 --num_iter 20 --long_sent  

    You should be able to observe the sentences getting longer. At the end, the directory should have a long_lengths.jpg with the graph of lengths over time, and a long_sent_20.pt as the saved model.
    
    * Train to produce sentences with valid words:
        python3 reinforce.py --num_episodes 5 --num_iter 30 --validity 

    You should be able to observe the sentences having less and less invalid words. At the end, the directory should have a validity.jpg with the graph of average sentence validity over time. The sentences get more repetitive over time.

    * Train to produce sentences with valid words without being repetitive (using the weighted reward function):
        python3 reinforce.py --num_episodes 5 --num_iter 8 --seed 14

    You should observe the sentences having fewer invalid words without being repetitive. Empirically we have found that 8 iterations is enough most of the times for the model to produce decent sentences. 

4. If you want to generate sentences from a trained model, do
        python3 generate.py --checkpoint [model_name].pt

Other:
The exploration directory contains our experiments with training a feed forward neural network with the Amazon Appliances Question Answer dataset. To train this network, do:
        python -m spacy download en_core_web_md
        python3 reinforce.py qs.txt

Notes:
    The Word RNN was trained using the implementation from Pytorch's Github:
    https://github.com/pytorch/examples/tree/master/word_language_model






        