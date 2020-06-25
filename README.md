# chatbot
Relatively simple AI-based chatbot that simulates human conversation through text chats. <br/>
His name is Tony and he is the new meme lord. Elon is retiring :D

## Neural network architecture
Used neural network is fairly standard feed-forward neural network with 2 hidden layers per 8 neurons.<br />

- Inputs to the network are one hot encoded values of "bag of words" - bag of words because the order in<br />
which the words appear in the sentence is lost, we only know the presence of words in the models vocabulary.<br />

Each sentence is represented with a list the length of the amount of words in models vocabulary.<br />
Each position in the list will represent a word from the vocabulary.<br/>
If the position in the list is a 1 then that will mean that the word exists in input sentence, if it is a 0 then the word is nor present.<br/>

- Similarly to a bag of words output lists which are the length of the amount of labels we have in our dataset.<br /> 
Each position in the list will represent one distinct label, a 1 in any of those positions will show which label is represented.
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
