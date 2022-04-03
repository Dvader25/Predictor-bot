import pandas_datareader._utils
import os
import weatherscraper as ws
import stock
import discord

data = 'Please enter a valid city name'
client = discord.Client()


@client.event
async def on_ready():
    print('Logged in')


@client.event
async def on_message(message):
    global data
    if message.content.startswith('!weather'):
        list = message.content.split()
        try:
            data = ws.construct(list[1], int(list[2]), "caa2c92ea7mshccadecb41402530p1df860jsn3d0dc6ed49fa")


        except KeyError:
            await message.channel.send(f"{list[1]} is not a valid name")
        finally:
            await message.channel.send(data)

    elif message.content.startswith('!predict'):
        list2 = message.content.split()
        await message.channel.send('Training')
        try:
            predictor = stock.training(list2[1], stock.start_date1, stock.end_date1)

        except pandas_datareader._utils.RemoteDataError :
            await message.channel.send('Please enter a valid stock name')

        try:
            await message.channel.send(file=discord.File('plot.png'))
            os.remove('plot.png')
            await message.channel.send(f'predicted price : {predictor[0]}$')
        finally:
            pass

    elif message.content.startswith('!help') :
        await message.channel.send('Type !predict and the stock symbol for the price \n '
                                   'Type !weather, the city, and the desired temp is celsius for the times')






client.run('OTU5OTA1Mzc4MDkzNjQ1ODU0.YkirjA.fyBi_3f1YSHyHI6dqFOeHJybPZs')
