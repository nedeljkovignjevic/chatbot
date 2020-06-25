# chatbot
Relatively simple AI-based chatbot that simulates human conversation through text chats. <br/>
His name is Tony and he is the new meme lord. Elon is retiring :D

## Neural network architecture
For neural network, I'm using fairly standard feed-forward neural network with 2 hidden layers per 8 neurons.<br />

I'm representing each sentence with a list the length of the amount of words in models vocabulary.<br />
Each position in the list will represent a word from the vocabulary.<br/>
If the position in the list is a 1 then that will mean that the word exists in input sentence,<br />
if it is a 0 then the word is nor present. We can call this a bag of words because the order in which the words appear in the sentence is lost,<br />
we only know the presence of words in the models vocabulary.

As well as formatting the input output needs to make sense to the neural network. <br />
Similarly to a bag of words output lists which are the length of the amount of labels we have in our dataset.<br /> 
Each position in the list will represent one distinct label, a 1 in any of those positions will show which label/tag is represented.
<br/><br/>

## Requirements
You can use pip to install the following:
```
1. python 3
2. pytorch
3. numpy
4. nltk
```
<br /><br />