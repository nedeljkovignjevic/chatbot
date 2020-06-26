"""
MIT License

Copyright (c) 2020 Nedeljko Vignjević

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""

from src.bot import Bot
from src.model import Model, keras   # function pointer


if __name__ == '__main__':

    # Load neural network model
    model = keras.models.load_model('model/model.h5')

    # Create bot
    bot = Bot(model)

    # Run chat loop
    while True:
        inp = input("You: ")
        if inp.lower() == "q":
            break

        response = bot.respond(inp)
        print("chatbot: " + response)
