So I could not get the policy network to converge, it seemed to converge to specific words and not budge:
    like if I gave a string like "Another day of finals" it would output
        "Another day of finals out out out out out out" 

I didn't know what to make of it, but I suspect it might just be too simple for a broad generalization like NLP, so I used an RNN implementation from 
pytorch's github:
    https://github.com/pytorch/examples/tree/master/word_language_model

Anyway I think I got it working to make short and long sentences, this time the results actually hold consistently. I had to train the rnn-model myself (on the cluster) and it was easiest with their dataset so I haven't used the QA dataset yet. Github would not allow me to upload that model so I sent you the link to the model.pt through Messenger. 

    Model link: https://drive.google.com/file/d/15HVjUQfGYEdx0nuvk15x1E4aTbt8caG8/view

    Put this model.pt in master/word_language_model/ 
    Then run 
        python reinforce.py --temperature 1.2

    The script will load model.pt (make a copy of it) and then start training it. You should have a lengths.jpg file in the end that plots a nice graph.

    I am currently implementing other reward functions; i did the make long sentences one and i thought it could be cool to do a sentiment analysis using nltk and make the model output happier stuff. I havent finished that yet though.

    Let me know if shit doesn't work and where you think we should be headed next. 
