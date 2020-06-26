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
