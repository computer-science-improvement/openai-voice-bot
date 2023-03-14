import openai
import io
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import whisper
import requests
# from pydub import AudioSegment
import os
from dotenv import load_dotenv

load_dotenv()

VOICES = {
    'ADAM': 'pNInz6obpgDQGcFmaJgB',  # Adam (American, clear)
    'RACHEL': '21m00Tcm4TlvDq8ikWAM',  # Rachel (american, mellow)
    'DOMI': 'AZnzlk1XvdvUeBnXmlld',  # Domi (american, engaged)
    'BELLA': 'EXAVITQu4vr4xnSDxMaL',  # Bella (American, soft)
    'ANTONI': 'ErXwobaYiN019PkySvjV',  # Antoni (American, modulated)
    'ELLI': 'MF3mGyEYCl7XYWbV9V6O',  # Elli (american, clear)
    'JOSH': 'TxGEqnHWrfWFTfGW9XjX',  # Josh (american, silvery)
    'ARNOLD': 'VR6AewLTigWG4xSOukaG',  # Arnold (american, nasal)
    'SAM': 'yoZ06aMxZJJ28mfd3POQ',  # Sam (american, dynamic)
}

ELEVENLABS_ENDPOINTS = {
    'TEXT_TO_SPEECH': f"https://api.elevenlabs.io/v1/text-to-speech/{VOICES['ADAM']}"
}

print(os.getenv('OPENAI_API_KEY'))

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

openai.api_key = OPENAI_API_KEY

bot = Bot(TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)


async def get_voice(prompt):
    headers = {
        'accept': 'audio/opus',
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        'text': prompt,
    }

    response = requests.post(ELEVENLABS_ENDPOINTS['TEXT_TO_SPEECH'], headers=headers, json=payload)
    # with open('prompt_response.ogg', 'wb') as f:
    #     f.write(response.content)
    # sound = AudioSegment.from_mp3('prompt_response.mp3')
    # sound.export('prompt_response.ogg', format='ogg', bitrate='32k')
    buffer_audio = io.BytesIO(response.content)
    return buffer_audio


async def get_openai_response(prompt):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=0.9,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=['You:']
    )

    return response['choices'][0]['text'].strip()


async def audio_2_text(audio_link):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_link)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    result = model.transcribe(audio_link, fp16=False)

    return result['text']


@dp.message_handler(content_types='voice')
async def voice_send(message: types.Message):
    voice = message.voice
    file_id = voice.file_id

    print(voice)

    audio_file_meta = await bot.get_file(file_id)
    file_path = audio_file_meta.file_path
    audio_link = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

    text_result = await audio_2_text(audio_link)
    prompt_response = await get_openai_response(text_result)

    print(message)
    print(message.voice)

    chat_id = message.chat.id

    print(message.text)

    await message.reply(text_result)

    voice_answer = await get_voice(prompt_response)
    await bot.send_voice(chat_id=chat_id, voice=voice_answer)
    await message.answer(prompt_response)


@dp.message_handler(content_types='text')
async def text_send(message: types.Message):
    separator = 'IMAGE'
    prompt = message.text.split(separator)[1]

    if 'IMAGE' in message.text:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size='256x256'
        )

        response_url = response['data'][0]['url']
        await message.reply_photo(photo=response_url)
    else:
        if prompt:
            text_result = await get_openai_response(prompt)
            await message.reply(text_result)
        else:
            await message.reply('[INVALID PROMPT]')


executor.start_polling(dp, skip_updates=True)
