import asyncio
import logging
import requests
import os
import re
import warnings
warnings.filterwarnings('ignore')
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, \
    CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)
load_dotenv()

API_TOKEN = os.getenv('API_TOKEN')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GIS_API_KEY = os.getenv('GIS_API_KEY')
GIGACHAT_API_KEY = os.getenv('GIGACHAT_API_KEY')

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

giga = GigaChat(credentials=GIGACHAT_API_KEY, verify_ssl_certs=False)

class HuggingFaceSentimentAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.labels = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'POSITIVE',
            'LABEL_2': 'NEUTRAL'
        }
        self.load_model()

    def load_model(self):
        try:
            model_name = "blanchefort/rubert-base-cased-sentiment"
            logging.info(f"Загрузка модели {model_name}...")

            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )

            logging.info(f"Модель успешно загружена на {self.device}")
            return True

        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            try:
                model_name = "seara/rubert-tiny2-russian-sentiment"
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.labels = {
                    'positive': 'POSITIVE',
                    'negative': 'NEGATIVE',
                    'neutral': 'NEUTRAL'
                }
                logging.info(f"Альтернативная модель загружена")
                return True
            except Exception as e2:
                logging.error(f"Ошибка загрузки альтернативной модели: {e2}")
                self.pipeline = None
                return False

    def classify(self, text):
        if not text or len(text.strip()) < 3:
            return {"tonality": "NEUTRAL", "confidence": 0.5}

        if self.pipeline is None:
            return {"tonality": "NEUTRAL", "confidence": 0.5}

        try:
            if len(text) > 512:
                text = text[:512]

            result = self.pipeline(text)[0]

            label = result['label']
            score = result['score']

            if label.lower() in self.labels:
                tonality = self.labels[label.lower()]
            elif label.upper() in self.labels:
                tonality = self.labels[label.upper()]
            else:
                if 'positive' in label.lower() or 'pos' in label.lower():
                    tonality = 'POSITIVE'
                elif 'negative' in label.lower() or 'neg' in label.lower():
                    tonality = 'NEGATIVE'
                else:
                    tonality = 'NEUTRAL'

            confidence = round(float(score), 2)

            return {
                "tonality": tonality,
                "confidence": confidence,
                "raw_label": label
            }

        except Exception as e:
            logging.error(f"Ошибка при анализе тональности: {e}")
            return {"tonality": "NEUTRAL", "confidence": 0.5}


sentiment_analyzer = HuggingFaceSentimentAnalyzer()

def classify_sentiment(text):
    try:
        return sentiment_analyzer.classify(text)
    except Exception as e:
        logging.error(f"Ошибка в classify_sentiment: {e}")
        return {"tonality": "NEUTRAL", "confidence": 0.5}

class AddressState(StatesGroup):
    waiting_for_city = State()
    waiting_for_street = State()
    waiting_for_house = State()

class SearchState(StatesGroup):
    waiting_for_selection = State()
    waiting_for_location = State()

SEARCH_CATEGORIES = {
    "coffee": {
        "name": "Попить кофейку",
        "queries": ["кофейня", "кофе", "кофе с собой"],
        "description": "кофейни"
    },
    "food": {
        "name": "Покушать",
        "queries": ["ресторан", "кафе", "столовая", "еда"],
        "description": "рестораны и кафе"
    },
    "party": {
        "name": "Оторваться от души",
        "queries": ["клуб", "ночной клуб", "антикафе", "кальянная", "бар"],
        "description": "места для отдыха"
    },
    "kids": {
        "name": "Для детей",
        "queries": ["батут", "детский центр", "зоопарк", "цирк", "детское кафе"],
        "description": "детские развлечения"
    },
    "date": {
        "name": "Свидание/Прогулка",
        "queries": ["парк", "кинотеатр", "театр", "музей", "аттракционы"],
        "description": "места для свиданий"
    }
}

user_selections = {}

def clean_gigachat_response(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_main_keyboard():
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Отправить геолокацию", request_location=True)],
            [KeyboardButton(text="Ввести адрес")],
            [KeyboardButton(text="Прогуляться бы...")],
            [KeyboardButton(text="Помощь")]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выбери действие..."
    )
    return kb


def get_cancel_keyboard():
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Отмена")]
        ],
        resize_keyboard=True
    )
    return kb


def get_location_request_keyboard():
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Отправить геолокацию", request_location=True)],
            [KeyboardButton(text="Отмена")]
        ],
        resize_keyboard=True
    )
    return kb


def get_category_keyboard(selected_categories=None):
    if selected_categories is None:
        selected_categories = []

    buttons = []

    for key, category in SEARCH_CATEGORIES.items():
        prefix = "✅ " if key in selected_categories else "⬜ "
        text = f"{prefix}{category['name']}"
        buttons.append([InlineKeyboardButton(text=text, callback_data=f"cat_{key}")])

    buttons.append([
        InlineKeyboardButton(text="Найти места", callback_data="search_confirm"),
        InlineKeyboardButton(text="Очистить все", callback_data="search_clear")
    ])
    buttons.append([InlineKeyboardButton(text="Отмена", callback_data="search_cancel")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def fetch_weather_by_coords(lat, lon, location_name):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            desc = data['weather'][0]['description']
            return (f"Погода в {location_name}:\n"
                    f"Температура: {temp}°C (ощущается как {feels_like}°C)\n"
                    f"Влажность: {humidity}%\n"
                    f"Ветер: {wind_speed} м/с\n"
                    f"{desc.capitalize()}")
        return "Не удалось получить данные о погоде."
    except Exception as e:
        return f"Ошибка соединения при получении погоды: {e}"


def get_nearby_places(lat, lon, query="кофейня", radius=1000):
    url = "https://catalog.api.2gis.com/3.0/items"
    params = {
        "q": query,
        "point": f"{lon},{lat}",
        "radius": radius,
        "key": GIS_API_KEY,
        "fields": "items.point,items.address_name,items.rubrics",
        "limit": 5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            items = response.json().get('result', {}).get('items', [])
            if not items:
                return f"{query.capitalize()} не найдены в радиусе {radius}м."

            result_text = []
            for i, item in enumerate(items[:3], 1):
                name = item.get('name', 'Без названия')
                address = item.get('address_name', 'Адрес не указан')
                result_text.append(f"{i}. {name}\n   {address}")

            return "\n\n".join(result_text)
        return f"Ошибка поиска {query}."
    except Exception as e:
        return f"Ошибка соединения с сервисом карт: {e}"


def reverse_geocode_2gis(lat, lon):
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


def geocode_address_2gis(city, street="", house=""):
    if street and house:
        query = f"{city}, {street}, {house}"
    else:
        query = city

    url = "https://catalog.api.2gis.com/3.0/items"
    params = {
        "q": query,
        "key": GIS_API_KEY,
        "fields": "items.point",
        "limit": 1
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if 'result' in data and data['result'].get('items'):
            point = data['result']['items'][0]['point']
            return float(point['lat']), float(point['lon'])
    except Exception as e:
        logging.error(f"Ошибка геокодирования адреса: {e}")
    return None

def search_places_by_categories(lat, lon, categories):
    results = []
    for cat_key in categories:
        if cat_key in SEARCH_CATEGORIES:
            category = SEARCH_CATEGORIES[cat_key]
            query = category['queries'][0]
            places = get_nearby_places(lat, lon, query=query, radius=1000)
            results.append(f"{category['name']}:\n{places}")
    return "\n\n".join(results) if results else "Ничего не найдено."

def find_parks_near_location(lat, lon, radius=1000):
    parks = get_nearby_places(lat, lon, query="парк", radius=radius)
    squares = get_nearby_places(lat, lon, query="сквер", radius=radius)
    gardens = get_nearby_places(lat, lon, query="сад", radius=radius)

    result = []
    if parks and "Не найдены" not in parks:
        result.append(f"Парки:\n{parks}")
    if squares and "Не найдены" not in squares:
        result.append(f"Скверы:\n{squares}")
    if gardens and "Не найдены" not in gardens:
        result.append(f"Сады:\n{gardens}")

    return "\n\n".join(result) if result else "В этом районе не найдено парков."

@dp.message(Command('start'))
async def send_welcome(message: Message):
    welcome_text = (
        "Привет! Я твой помощник по поиску мест и прогулкам в Уфе!\n\n"
        "Нажми 'Отправить геолокацию' - покажу погоду и кофейни рядом\n"
        "'Ввести адрес' - введи адрес вручную\n"
        "'Прогуляться бы...' - выбери категории для поиска\n"
        "'Помощь' - список всех команд\n\n"
        "Или просто задай любой вопрос - я отвечу!\n\n"
        "Важная функция: Я анализирую тональность твоих сообщений и подстраиваю ответы под твоё настроение!"
    )
    await message.answer(welcome_text, parse_mode="HTML", reply_markup=get_main_keyboard())

@dp.message(Command('help'))
@dp.message(F.text == "Помощь")
async def send_help(message: Message):
    help_text = (
        "Доступные функции:\n\n"
        "Отправить геолокацию - покажу погоду и кофейни рядом\n"
        "Ввести адрес - пошаговый ввод адреса\n"
        "Прогуляться бы... - выбор категорий для поиска:\n"
        "   • Попить кофейку\n"
        "   • Покушать\n"
        "   • Оторваться от души\n"
        "   • Для детей\n"
        "   • Свидание/Прогулка\n\n"
        "Общение:\n"
        "Просто напиши любой вопрос - я отвечу как ИИ-помощник\n\n"
        "Анализ тональности:\n"
        "Я анализирую настроение в твоих сообщениях и показываю результат!\n"
        "Попробуй написать что-то позитивное или негативное, чтобы увидеть разницу.\n\n"
        "Команды:\n"
        "/start - перезапуск бота\n"
        "/cancel - отмена текущей операции\n"
        "/help - это сообщение\n"
        "/sentiment - проверить тональность текста"
    )
    await message.answer(help_text, parse_mode="HTML", reply_markup=get_main_keyboard())


@dp.message(Command('sentiment'))
async def test_sentiment(message: Message):
    text = message.text.replace('/sentiment', '').strip()
    if not text:
        await message.answer("Напиши текст после команды, например:\n/sentiment Я люблю Уфу!")
        return

    result = classify_sentiment(text)

    response = (
        f"Анализ тональности:\n\n"
        f"Текст: {text}\n\n"
        f"Тональность: {result['tonality']}\n"
        f"Уверенность: {result['confidence'] * 100}%\n"
        f"Сырая метка: {result.get('raw_label', 'N/A')}"
    )
    await message.answer(response, parse_mode="HTML")


@dp.message(Command('cancel'))
async def cancel_handler(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id in user_selections:
        del user_selections[user_id]
    await state.clear()
    await message.answer("Операция отменена.", reply_markup=get_main_keyboard())


@dp.message(F.text == "Ввести адрес")
async def start_address_input(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Введите название города:",
        reply_markup=get_cancel_keyboard()
    )
    await state.set_state(AddressState.waiting_for_city)


@dp.message(AddressState.waiting_for_city)
async def process_city(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return
    await state.update_data(city=message.text.strip())
    await message.answer("Введите название улицы:")
    await state.set_state(AddressState.waiting_for_street)


@dp.message(AddressState.waiting_for_street)
async def process_street(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return
    await state.update_data(street=message.text.strip())
    await message.answer("Введите номер дома:")
    await state.set_state(AddressState.waiting_for_house)


@dp.message(AddressState.waiting_for_house)
async def process_house(message: Message, state: FSMContext):
    if message.text == "Отмена":
        await cancel_handler(message, state)
        return

    house = message.text.strip()
    data = await state.get_data()
    city = data.get('city')
    street = data.get('street')

    if not city or not street:
        await message.answer("Ошибка: данные города или улицы не найдены. Начните заново.")
        await state.clear()
        return

    status_msg = await message.answer(f"Ищу данные для: {city}, {street}, {house}...")

    coords = geocode_address_2gis(city, street, house)

    if coords:
        lat, lon = coords
        await status_msg.delete()
        weather = fetch_weather_by_coords(lat, lon, f"{city}, {street}")
        cafes = get_nearby_places(lat, lon, query="кофейня")
        response_text = f"{weather}\n\nБлижайшие кофейни:\n{cafes}"
        await message.answer(response_text, reply_markup=get_main_keyboard())
    else:
        await status_msg.edit_text(
            "Адрес не найден. Проверьте корректность данных.\n"
            "Попробуйте еще раз через 'Ввести адрес'"
        )
    await state.clear()


@dp.message(F.text == "Прогуляться бы...")
async def walk_button_pressed(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_selections[user_id] = []
    await state.set_state(SearchState.waiting_for_selection)
    await message.answer(
        "Выбери, что именно хочешь найти:\n"
        "(можно выбрать несколько вариантов)\n"
        "После выбора нажми 'Найти места'",
        parse_mode="HTML",
        reply_markup=get_category_keyboard([])
    )


@dp.callback_query(lambda c: c.data and c.data.startswith('cat_'))
async def process_category_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    category_key = callback.data.replace('cat_', '')
    selected = user_selections.get(user_id, [])

    if category_key in selected:
        selected.remove(category_key)
    else:
        selected.append(category_key)

    user_selections[user_id] = selected

    try:
        await callback.message.edit_text(
            "Выбери, что именно хочешь найти:\n"
            "(можно выбрать несколько вариантов)\n"
            "После выбора нажми 'Найти места'",
            parse_mode="HTML",
            reply_markup=get_category_keyboard(selected)
        )
    except Exception:
        pass

    await callback.answer()


@dp.callback_query(lambda c: c.data == "search_clear")
async def clear_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    user_selections[user_id] = []

    try:
        await callback.message.edit_text(
            "Выбери, что именно хочешь найти:\n"
            "(можно выбрать несколько вариантов)\n"
            "После выбора нажми 'Найти места'",
            parse_mode="HTML",
            reply_markup=get_category_keyboard([])
        )
    except Exception:
        pass

    await callback.answer()


@dp.callback_query(lambda c: c.data == "search_cancel")
async def cancel_search(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    if user_id in user_selections:
        del user_selections[user_id]

    await state.clear()

    try:
        await callback.message.delete()
    except Exception:
        pass

    await callback.message.answer(
        "Поиск отменен.",
        reply_markup=get_main_keyboard()
    )
    await callback.answer()


@dp.callback_query(lambda c: c.data == "search_confirm")
async def confirm_search(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    selected = user_selections.get(user_id, [])

    if not selected:
        await callback.answer("Выбери хотя бы одну категорию!", show_alert=True)
        return

    await state.update_data(search_categories=selected)
    await state.set_state(SearchState.waiting_for_location)

    await callback.message.delete()

    await callback.message.answer(
        "Отлично! Теперь отправь геолокацию или нажми 'Отмена'",
        parse_mode="HTML",
        reply_markup=get_location_request_keyboard()
    )
    await callback.answer()


@dp.message(SearchState.waiting_for_location, F.location)
async def search_with_location(message: Message, state: FSMContext):
    lat, lon = message.location.latitude, message.location.longitude

    data = await state.get_data()
    selected_categories = data.get('search_categories', [])

    status_msg = await message.answer("Ищу места по твоему запросу...")

    city, address = reverse_geocode_2gis(lat, lon)
    places = search_places_by_categories(lat, lon, selected_categories)

    await status_msg.delete()

    selected_names = [f"{SEARCH_CATEGORIES[cat]['name']}"
                      for cat in selected_categories]
    categories_text = ", ".join(selected_names)

    location_display = f"{city}, {address}" if city != "Неизвестный город" else f"координаты: {lat:.4f}, {lon:.4f}"

    response_text = (
        f"Местоположение: {location_display}\n"
        f"Искал: {categories_text}\n\n"
        f"{places}"
    )

    await message.answer(response_text, parse_mode="HTML", reply_markup=get_main_keyboard())
    await state.clear()

    user_id = message.from_user.id
    if user_id in user_selections:
        del user_selections[user_id]


@dp.message(SearchState.waiting_for_location, F.text == "Отмена")
async def cancel_location_search(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id in user_selections:
        del user_selections[user_id]

    await state.clear()
    await message.answer(
        "Поиск отменен.",
        reply_markup=get_main_keyboard()
    )


async def process_location_info(message: Message, lat: float, lon: float, location_name: str, search_type="general"):
    weather = fetch_weather_by_coords(lat, lon, location_name)

    if search_type == "walk":
        parks = find_parks_near_location(lat, lon)
        cafes = get_nearby_places(lat, lon, query="кофейня")
        response_text = f"{weather}\n\nМеста для прогулки:\n{parks}\n\nКофейни рядом:\n{cafes}"
    else:
        cafes = get_nearby_places(lat, lon, query="кофейня")
        response_text = f"{weather}\n\nБлижайшие кофейни:\n{cafes}"

    await message.answer(response_text, reply_markup=get_main_keyboard())


@dp.message(F.location)
async def handle_location(message: Message):
    lat, lon = message.location.latitude, message.location.longitude
    city, address = reverse_geocode_2gis(lat, lon)
    location_display = f"{city}, {address}"
    await process_location_info(message, lat, lon, location_display)


@dp.message(F.text & ~F.commands)
async def chat_with_ai(message: Message, state: FSMContext):
    current_state = await state.get_state()

    if current_state is not None:
        return

    if message.text in ["Отправить геолокацию", "Ввести адрес", "Прогуляться бы...", "Помощь", "Отмена"]:
        return

    try:
        tonal_result = classify_sentiment(message.text)
        tonality_label = tonal_result["tonality"]
        confidence_score = tonal_result["confidence"]

        logging.info(f"Тональность: {tonality_label} (уверенность: {confidence_score})")

        print(
            f"Тональность: {tonality_label} ({confidence_score}) - '{message.text[:50]}...'")

    except Exception as e:
        logging.error(f"Ошибка анализа тональности: {e}")
        tonality_label = "NEUTRAL"
        confidence_score = 0.5

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        if tonality_label == "POSITIVE" and confidence_score > 0.7:
            tone_prompt = "Пользователь в хорошем настроении, ответь с энтузиазмом."
        elif tonality_label == "NEGATIVE" and confidence_score > 0.7:
            tone_prompt = "Пользователь расстроен, ответь с сочувствием и пониманием, предложи помощь."
        else:
            tone_prompt = "Ответь дружелюбно и информативно."

        system_prompt = f"""Ты дружелюбный помощник-гид по городу Уфа. 
Отвечай кратко и по делу.
Знаешь все достопримечательности, кафе, парки и развлечения Уфы.
{tone_prompt}"""

        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=system_prompt
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=message.text
                )
            ],
            temperature=0.7,
            max_tokens=500
        )

        response = giga.chat(payload)

        if response and response.choices:
            answer = response.choices[0].message.content
            clean_answer = clean_gigachat_response(answer)

            confidence_percent = int(confidence_score * 100)

            sentiment_info = (
                f"Анализ тональности вашего сообщения:\n"
                f"{tonality_label} (уверенность: {confidence_percent}%)\n\n"
            )

            final_answer = f"{sentiment_info}{clean_answer}"

            await message.answer(final_answer, parse_mode="HTML", reply_markup=get_main_keyboard())
        else:
            await message.answer("Извините, не удалось получить ответ. Попробуйте позже.",
                                 reply_markup=get_main_keyboard())

    except Exception as e:
        logging.error(f"Ошибка GigaChat: {e}")
        await message.answer(
            "Я здесь! Если хочешь найти интересные места в Уфе, воспользуйся кнопками меню:\n\n"
            "Отправить геолокацию\n"
            "Ввести адрес\n"
            "Прогуляться бы...",
            reply_markup=get_main_keyboard()
        )


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Бот с GigaChat и Hugging Face запущен!")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())