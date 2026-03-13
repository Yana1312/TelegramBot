import asyncio
import logging
import requests
import os
import json
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GIS_API_KEY = os.getenv('GIS_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1"
)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


class AddressState(StatesGroup):
    waiting_for_city = State()
    waiting_for_street = State()
    waiting_for_house = State()


def fetch_weather_by_coords(lat: float, lon: float, location_name: str) -> str:
    """Получение погоды по координатам"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"Погода в {location_name}: {temp} градусов, {desc}."
        return "Не удалось получить данные о погоде."
    except Exception as e:
        return f"Ошибка соединения при получении погоды: {e}"


def get_nearby_places(lat, lon, query="кофейня", radius=1000):
    """Поиск заведений через 2GIS"""
    url = "https://catalog.api.2gis.com/3.0/items"
    params = {
        "q": query,
        "point": f"{lon},{lat}",
        "radius": radius,
        "key": GIS_API_KEY,
        "fields": "items.address_name"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get('result', {}).get('items', [])
            if not items:
                return "Места по вашему запросу не найдены."
            return "\n\n".join([f"{i.get('name')}\nАдрес: {i.get('address_name', 'не указан')}" for i in items[:3]])
        return "Ошибка поиска заведений."
    except Exception as e:
        return f"Ошибка соединения с сервисом карт: {e}"


def reverse_geocode_2gis(lat: float, lon: float) -> tuple:
    """Определение адреса по координатам"""
    try:
        url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "point": f"{lon},{lat}",
            "radius": 1000,
            "key": GIS_API_KEY,
            "fields": "items.city,items.address_name,items.subtype,items.type",
            "limit": 5
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("result", {}).get("items", [])
            city = "Неизвестный город"
            address = "Адрес не определен"
            if items:
                city_item = next((i for i in items if i.get('subtype') == 'city'), None)
                city = city_item.get('name') if city_item else items[0].get('name', city)

                addr_item = next((i for i in items if i.get('address_name')), None)
                address = addr_item.get('address_name') if addr_item else address
            return city, address
    except Exception as e:
        logging.error(f"Ошибка геокодирования: {e}")
    return "Неизвестный город", "Адрес не определен"

def geocode_address_2gis(city: str, street: str, house: str):
    """Преобразование адреса в координаты"""
    query = f"{city}, {street}, {house}"
    url = "https://catalog.api.2gis.com/3.0/items"
    params = {"q": query, "key": GIS_API_KEY, "fields": "items.point", "limit": 1}
    try:
        res = requests.get(url, params=params).json()
        point = res['result']['items'][0]['point']
        return float(point['lat']), float(point['lon'])
    except:
        return None


def analyze_query_with_ai(text):
    """Анализ текста через DeepSeek для извлечения параметров"""
    prompt = f"Извлеки из запроса: '{text}'. Верни JSON объект с полями: city (город), radius (число в метрах), categories (список категорий для поиска)."
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # Модель DeepSeek
            messages=[
                {"role": "system", "content": "Вы работаете как парсер данных. Ответ должен содержать только валидный JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Ошибка DeepSeek: {e}")
        return None

@dp.message(Command('start'))
async def send_welcome(message: Message):
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Отправить локацию", request_location=True)],
            [KeyboardButton(text="Ввести адрес вручную")],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True
    )
    await message.answer(
        "Вы запустили систему навигации. Отправьте геолокацию или введите адрес для получения информации.",
        reply_markup=kb
    )


@dp.message(Command('help'))
@dp.message(F.text == "Помощь")
async def send_help(message: Message):
    help_text = (
        "Доступные функции:\n\n"
        "1. Геолокация: расчет погоды и поиск кофеен в текущем районе.\n"
        "2. Ввод адреса: пошаговое указание города и улицы для поиска мест.\n"
        "3. Текстовый запрос: ввод названия города или сложного описания для анализа системой.\n\n"
        "Список команд:\n"
        "/start - запуск бота\n"
        "/address - ввод адреса вручную\n"
        "/cancel - отмена текущей операции"
    )
    await message.answer(help_text, parse_mode="HTML")


@dp.message(Command('cancel'))
async def cancel_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Операция отменена.")



@dp.message(Command('address'))
@dp.message(F.text == "Ввести адрес вручную")
async def start_address_input(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Введите название города.")
    await state.set_state(AddressState.waiting_for_city)


@dp.message(AddressState.waiting_for_city)
async def process_city(message: Message, state: FSMContext):
    await state.update_data(city=message.text.strip())
    await message.answer("Введите название улицы.")
    await state.set_state(AddressState.waiting_for_street)


@dp.message(AddressState.waiting_for_street)
async def process_street(message: Message, state: FSMContext):
    await state.update_data(street=message.text.strip())
    await message.answer("Введите номер дома.")
    await state.set_state(AddressState.waiting_for_house)


@dp.message(AddressState.waiting_for_house)
async def process_house(message: Message, state: FSMContext):
    house = message.text.strip()
    data = await state.get_data()
    city, street = data['city'], data['street']

    await message.answer(f"Поиск данных для: {city}, {street}, {house}")
    coords = geocode_address_2gis(city, street, house)

    if coords:
        lat, lon = coords
        await state.clear()
        await process_location_info(message, lat, lon, city)
    else:
        await message.answer("Адрес не найден. Проверьте корректность данных.")
        await state.clear()


# --- Обработка локации и текста ---
async def process_location_info(message: Message, lat: float, lon: float, location_name: str):
    weather = fetch_weather_by_coords(lat, lon, location_name)
    cafes = get_nearby_places(lat, lon, query="кофейня")
    response_text = f"{weather}\n\nРезультаты поиска заведений:\n{cafes}"
    await message.answer(response_text)


@dp.message(F.location)
async def handle_location(message: Message):
    lat, lon = message.location.latitude, message.location.longitude
    city, address = reverse_geocode_2gis(lat, lon)
    location_display = f"{city}, {address}"
    await process_location_info(message, lat, lon, location_display)


@dp.message(F.text & ~F.commands)
async def handle_text_logic(message: Message, state: FSMContext):
    if await state.get_state():
        return

    if message.text in ["Отправить локацию", "Ввести адрес вручную", "Помощь"]:
        return

    if len(message.text.split()) > 3:
        wait_msg = await message.answer("Выполняется анализ запроса.")
        params = analyze_query_with_ai(message.text)

        if params and params.get('city'):
            lat, lon = geocode_address_2gis(params['city'], "", "")
            if lat:
                results = []
                for cat in params.get('categories', ['интересное место']):
                    found = get_nearby_places(lat, lon, query=cat, radius=params.get('radius', 1000))
                    results.append(f"Категория {cat}:\n{found}")

                await wait_msg.edit_text(f"Результаты для {params['city']}:\n\n" + "\n\n".join(results))
            else:
                await wait_msg.edit_text("Не удалось определить координаты указанного места.")
        else:
            await wait_msg.edit_text("Не удалось распознать параметры запроса.")
    else:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={message.text}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                desc = data['weather'][0]['description']
                await message.answer(f"Погода в {data['name']}: {temp} градусов, {desc}.")
            else:
                await message.answer("Город не найден.")
        except:
            await message.answer("Произошла ошибка при поиске города.")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Сервис запущен.")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())