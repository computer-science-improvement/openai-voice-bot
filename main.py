import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import whisper
import requests
# from pydub import AudioSegment
import os

ELEVENLABS_ENDPOINTS = {
    'TEXT_TO_SPEECH': 'https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB'
}

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

    json = {
        'text': prompt,
    }

    response = requests.post(ELEVENLABS_ENDPOINTS['TEXT_TO_SPEECH'], headers=headers, json=json)
    with open('prompt_response.ogg', 'wb') as f:
        f.write(response.content)
    # sound = AudioSegment.from_mp3('prompt_response.mp3')
    # sound.export('prompt_response.ogg', format='ogg', bitrate='32k')


@dp.message_handler(content_types='voice')
async def voice_send(message: types.Message):
    voice = message.voice
    file_id = voice.file_id

    print(voice)

    audio_file_meta = await bot.get_file(file_id)
    file_path = audio_file_meta.file_path
    audio_link = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    #
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_link)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    result = model.transcribe(audio_link, fp16=False)

    text_result = result['text']

    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=text_result,
        temperature=0.9,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=['You:']
    )

    prompt_response = response['choices'][0]['text'].strip()
    # print(response['choices'])
    print(message)
    print(message.voice)

    chat_id = message.chat.id

    print(message.text)

    await message.reply(text_result)

    await get_voice(prompt_response)
    await bot.send_voice(chat_id=chat_id, voice=open('prompt_response.ogg', 'rb'))
    await message.answer(prompt_response)


executor.start_polling(dp, skip_updates=True)
