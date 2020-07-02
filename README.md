# chatbot
Relatively simple AI-based chatbot that simulates human conversation through text chats.<br />
Most common use case for this type of chatbot is answering frequently asked questions. Also it can be used for fun.<br />


#### Chat example<br />
![fotka](https://user-images.githubusercontent.com/54076398/85853585-0e969500-b7b3-11ea-92ee-d193637dd0df.jpg)
<br /><br />

## Neural network architecture
Neural network model is fairly standard feed-forward neural network model with 2 hidden layers per 4 neurons.<br />

- Inputs to the network are one hot encoded values of "bag of words" - bag of words because the order in<br />
which the words appear in the sentence is lost, we only know the presence of words in the models vocabulary.<br />
Each sentence is represented with a list the length of the amount of words in models vocabulary.<br />
Each position in the list will represent a word from the vocabulary.<br/>
If the position in the list is a 1 then that will mean that the word exists in input sentence, if it is a 0 then the word is nor present.<br/>

- Similarly to a bag of words on output of neural network we have lists which are the length of <br /> 
the amount of labels we have in our dataset.<br /> 
Each position in the list will represent one distinct label, a 1 in any of those positions will show which label is represented.<br />
Labels: **greeting**, **goodbye**, **basic**, **thanks**, **meme**
<br/><br/>

### Requirements
You can use pip to install the following:
```
1. python 3
2. pytorch
3. numpy
4. nltk
```
<br />

## License
chatbot is released under the **Apache License, Version 2.0** (the "License");<br />
you may not use this file except in compliance with the License.<br />
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
