"""
Copyright (C) 2020 Nedeljko Vignjević

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import discord

from src.bot import Bot
from tensorflow import keras


client = discord.Client()


@client.event
async def on_ready():
    print('chatbot is ready.'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    await message.channel.send(bot.respond(message.content))


if __name__ == '__main__':

    # Load neural network model
    model = keras.models.load_model('model/model.h5')

    # Create bot
    bot = Bot(model)

    # Run client
    client.run('NzI4MDIwNDU3OTc4MjAwMDk1.Xv0oNg.zVbZRCsQbuVG_25YXNX2i8OUGWw')
