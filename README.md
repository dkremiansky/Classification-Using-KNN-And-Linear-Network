# Classification-Using-SVM-And-Neural-Network
project on the course "Natural Language Processing" (97215)

In this work, I implemented several models learned in class to solve the task of recognizing entities in the text (Named Entity Recognition), I performed language processing tasks on real data and analyzed the nature of my success.
In my work I implemented 2 models - an SVM model and a model that includes linear networks.

The data files are in the following format:
Each line represents a word, and contains the word and its label, separated by tabs. At the end of each sentence there is a blank line. We will refer to the labeling of the word as binary labeling - the labeling is negative if the word received the labeling O, and positive otherwise.
The training file is the file train.tagged, the test file is dev.tagged.

Each word is represented by a vector that consists of its representation in word2Vec and the representation of its environment. I implemented it in a vector that contains the word and the two words adjacent to it on each side.

I examined my results by looking at the F1 score.
Throughout the exercise, the F1 index refers to the binary F1 index at the word level. That is, chaining all the predications of your model and the real labels over all the sentences, and calculating binary F1 between the two lists.
