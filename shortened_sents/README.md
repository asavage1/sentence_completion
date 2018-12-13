Currently there are two .pt models in this directory: the normal shakespeare.pt that was trained on a corpus and a shakespeare_shortened.pt that was 'reinforced' to make short sentences.

To see the difference in the lengths of sentences (I modified the generate file to print the average length of 100 sentetences, and print the last sentence):
    python generate.py shakespeare.pt   # should give about ~65 word avg
    python generate.py shakespeare_shortened.pt # about ~22 word avg

Usage:
	If you want to train a model into emitting short sentences:
		python reinforce.py shakespeare.pt

I run it for only a couple of episodes right now, even at 10 episodes the model seems to spit out nonsense. Feel free to experiment. The graph plotted shows the lengths of the last sentence. If the graph plotted is a downwards sloping line then the model is fine to use. Sometimes the graph slopes up, which is when just 
        python reinforce.py shakespeare.pt
    again until the graph slopes down. I cannot explain the variance. 

    