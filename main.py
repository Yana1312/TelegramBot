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

def fetch_weather(url):
    """Запрос к OpenWeatherMap."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp, desc, city = data['main']['temp'], data['weather'][0]['description'], data['name']
            return f"В городе {city} сейчас {temp}°C, {desc}."
        return "Не удалось получить данные о погоде."
    except Exception as e:
        return f"Ошибка соединения (погода): {e}"


def get_nearby_cafes(lat, lon):
    """Поиск кофеен через 2GIS."""
    url = "https://catalog.api.2gis.com/3.0/items"
    params = {
        "q": "кофейня",
        "point": f"{lon},{lat}",
        "radius": 1000,
        "key": GIS_API_KEY,
        "fields": "items.address_name"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get('result', {}).get('items', [])
            if not items: return "Ничего не нашел поблизости."
            return "\n\n".join([f"{i.get('name')}\n {i.get('address_name', 'Адрес не указан')}" for i in items[:3]])
        return "Ошибка поиска в 2GIS."
    except Exception as e:
        return f"Ошибка соединения (2GIS): {e}"


@dp.message(F.text & ~F.commands)
async def handle_city_text(message: Message):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={message.text}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
    await message.answer(fetch_weather(url))

@dp.message(F.location)
async def handle_location(message: Message):
    lat, lon = message.location.latitude, message.location.longitude

    weather = fetch_weather(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru")
    places = get_nearby_cafes(lat, lon)

    final_text = (
        f"Локация получена!\n\n"
        f"{weather}\n\n"
        f"Ближайшие места:\n{places}"
    )
    await message.answer(final_text, parse_mode="Markdown")



async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())