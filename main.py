"""
MIT License

chatbot is relatively simple AI-based software application that simulates human conversation through text chats.
This file is part of chatbot.

Copyright (c) 2020 Nedeljko Vignjević

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""

from src.model import Model, ChatbotDataset
from src.bot import Bot


if __name__ == '__main__':

    # Used only to get input and output dimensions for model
    data_set = ChatbotDataset()

    # Prepare neural network model for evaluation
    model = Model(len(data_set.x[0]), len(data_set.y[0]))
    model.evaluate()

    # Create bot
    bot = Bot(model)

    # Run chat loop
    while True:
        inp = input("You: ")
        if inp.lower() == "q":
            break

        response = bot.respond(inp)
        print("chatbot: " + response)
