import asyncio
import logging
import requests
import os
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GIS_API_KEY = os.getenv('GIS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


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

            return f"Погода:\n{location_name}: {temp}°C, {desc}."
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
            if not items:
                return "Ничего не нашел поблизости."
            return "\n\n".join([f"{i.get('name')}\n {i.get('address_name', 'Адрес не указан')}" for i in items[:3]])
        return "Ошибка поиска в 2GIS."
    except Exception as e:
        return f"Ошибка соединения (2GIS): {e}"

def reverse_geocode_2gis(lat: float, lon: float) -> tuple:
    """Преобразует координаты в адрес"""
    try:
        url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "point": f"{lon},{lat}",
            "radius": 2000,
            "key": GIS_API_KEY,
            "fields": "items.city,items.address_name,items.name,items.full_name,items.subtype,items.type",
            "limit": 5
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 200:
            items = resp.json().get("result", {}).get("items", [])
            if items:
                city_item = next((i for i in items if i.get('subtype') == 'city'), None)
                if city_item:
                    city = city_item.get('name')
                else:
                    full_name_item = next((i for i in items if i.get('full_name')), None)
                    city = full_name_item['full_name'].split(',')[0].strip() if full_name_item else None

                if not city:
                    name_item = next((i for i in items if i.get('name')), None)
                    city = name_item.get('name') if name_item else "Неизвестный город"

                building = next((i for i in items if i.get('type') == 'building' and i.get('address_name')), None)
                if building:
                    address = building['address_name']
                else:
                    street = next((i for i in items if i.get('type') == 'street'), None)
                    address = street.get('name') if street else None

                if not address:
                    addr_item = next((i for i in items if i.get('address_name')), None)
                    address = addr_item['address_name'] if addr_item else None

                if not address:
                    name_item = next((i for i in items if i.get('name')), None)
                    address = name_item['name'] if name_item else "Адрес не определен"

                return city, address
    except Exception as e:
        logging.error(f"Ошибка геокодирования: {e}")

    return "Неизвестный город", "Адрес не определен"

def geocode_address_2gis(city: str, street: str, house: str):
    """Преобразует адрес в координаты"""
    address_query = f"{city}, {street}, {house}"
    try:
        url = "https://catalog.api.2gis.com/3.0/items"
        params = {
            "q": address_query,
            "key": GIS_API_KEY,
            "fields": "items.point",
            "limit": 1
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            items = resp.json().get("result", {}).get("items", [])
            if items:
                point = items[0].get("point", {})
                return float(point.get("lat")), float(point.get("lon"))
    except Exception as e:
        logging.error(f"Ошибка поиска адреса: {e}")
    return None

@dp.message(Command('start'))
async def send_welcome(message: Message):
    """Обновленный обработчик start"""
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Посоветуй место по локации", request_location=True)],
            [KeyboardButton(text="Ввести адрес")],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True
    )
    await message.answer(
        "Привет! Я твой умный гид.\n\n"
        "Нажми 'Посоветуй место по локации' и отправь геолокацию\n"
        "Или нажми 'Ввести адрес' для ручного ввода\n"
        "Можно просто написать название города!",
        reply_markup=kb
    )


@dp.message(Command('help'))
@dp.message(F.text == "Помощь")
async def send_help(message: Message):
    """Обновленный обработчик помощи"""
    help_text = (
        "<b>Я умею:</b>\n\n"
        "По геолокации:\n"
        "• Показывать погоду\n"
        "• Находить кофейни рядом\n\n"
        "По адресу (кнопка 'Ввести адрес'):\n"
        "• Пошаговый ввод города, улицы, дома\n"
        "• Поиск мест рядом\n\n"
        "Просто напиши название города - узнай погоду\n\n"
        "<b>Команды:</b>\n"
        "/start - перезапустить бота\n"
        "/address - ввести адрес\n"
        "/cancel - отменить ввод"
    )
    await message.answer(help_text, parse_mode="HTML")


@dp.message(Command('address'))
@dp.message(F.text == "Ввести адрес")
async def start_address_input(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Шаг 1: Введите название города (например, Москва).")
    await state.set_state(AddressState.waiting_for_city)


@dp.message(Command('cancel'))
async def cancel_handler(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Ввод отменен.")


@dp.message(AddressState.waiting_for_city)
async def process_city(message: Message, state: FSMContext):
    await state.update_data(city=message.text.strip())
    await message.answer("Шаг 2: Введите название улицы.")
    await state.set_state(AddressState.waiting_for_street)


@dp.message(AddressState.waiting_for_street)
async def process_street(message: Message, state: FSMContext):
    await state.update_data(street=message.text.strip())
    await message.answer("Шаг 3: Введите номер дома.")
    await state.set_state(AddressState.waiting_for_house)


async def process_location_info(message: Message, lat: float, lon: float, location_name: str):
    """Общая функция для обработки информации по координатам"""
    status_msg = await message.answer("Ищу информацию...")
    weather = fetch_weather_by_coords(lat, lon, location_name)
    cafes = get_nearby_cafes(lat, lon)

    await status_msg.delete()

    response_text = (
        f"{weather}\n\n"
        f"<b>Кофейни рядом:</b>\n{cafes}"
    )
    await message.answer(response_text, parse_mode="HTML")


@dp.message(AddressState.waiting_for_house)
async def process_house(message: Message, state: FSMContext):
    """обработчик ручного ввода адреса (адрес -> координаты)"""
    house = message.text.strip()
    data = await state.get_data()
    city, street = data['city'], data['street']

    status_msg = await message.answer(f"Ищу места около: {city}, {street}, {house}...")
    coords = geocode_address_2gis(city, street, house)

    if coords:
        lat, lon = coords
        await state.clear()
        await status_msg.delete()

        await process_location_info(message, lat, lon, city)
    else:
        await status_msg.edit_text("Адрес не найден. Проверьте данные и попробуйте снова.")
        await state.clear()


@dp.message(F.location)
async def handle_location(message: Message):
    """обработчик геолокации (координаты -> адрес)"""
    lat, lon = message.location.latitude, message.location.longitude

    logging.info(f"Координаты: {lat}, {lon}")

    city, address = reverse_geocode_2gis(lat, lon)

    logging.info(f"ИТОГ: city='{city}', address='{address}'")

    if city != "Местоположение" and address != f"координаты: {lat:.4f}, {lon:.4f}":
        location_display = f"{city}, {address}"
    else:
        location_display = f"{city}: {address}"

    logging.info(f"location_display='{location_display}'")

    await process_location_info(message, lat, lon, location_display)
@dp.message(F.text & ~F.commands)
async def handle_city_text(message: Message):
    if message.text in ["Посоветуй место по локации", "Ввести адрес", "Помощь"]:
        return

    city_name = message.text
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric&lang=ru"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            city_from_api = data['name']
            await message.answer(f"Погода в {city_from_api}: {temp}°C, {desc}.")
        else:
            await message.answer("Не удалось получить данные о погоде. Проверьте название города.")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")

async def main():
    if not API_TOKEN:
        logging.error("API_TOKEN не найден!")
        return
    if not WEATHER_API_KEY:
        logging.warning("WEATHER_API_KEY не найден - погода работать не будет")
    if not GIS_API_KEY:
        logging.warning("GIS_API_KEY не найден - поиск мест работать не будет")

    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())