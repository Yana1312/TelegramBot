import asyncio
import logging
import requests
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from openai import OpenAI
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GIS_API_KEY = os.getenv('GIS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)


@dp.message(Command('start'))
async def send_welcome(message: Message):
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Посоветуй место по локации", request_location=True)],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True
    )
    await message.answer("Привет! Я твой умный гид. Отправь локацию или напиши город!", reply_markup=kb)

@dp.message(Command('help'))
@dp.message(F.text == "Помощь")
async def send_help(message: Message):
    await message.answer(
        "Я умею:\n1. Показывать погоду.\n2. Находить кафе рядом.\n3. Давать советы через ИИ.\nПросто отправь мне свою геолокацию!")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())