import discord
import discord.message
from discord import on_message


intents = discord.Intents.default()
# intents.typing = False     ## bot의 typing 작업 필요 여부
# intents.presences = False  ## bot의 presence 작업 필요 여부

 # Somewhere else:
 # client = discord.Client(intents=intents)
 # or
 # from discord.ext import commands
 # bot = commands.Bot(command_prefix='!', intents=intents)

client = discord.Client()

class basic():
    help = '''
    ==help==
    사투리 TTS 봇 헬프 메세지입니다.
    아래 내용을 참고해주세요.
    '''


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
        
    if message.content.startwith('$help'):
        await message.channel.send(basic.help)
    
    if message.content.startwith('/TTS'):
        # 메세지에서 원하는 텍스트 가져오기
        # 가져와서 백엔드로 돌려서 음성 파일 제작
        # 만들어진 음성 파일을 디스코드를 통해 오디오 아웃풋

        

client.run('MTA3Mzg0ODQ3NDM4NDIxNjE2NA.GLzV5d.UhY4UuJqWhub1UjtcNYoV5qV6wrfwnNe_XG5_I')