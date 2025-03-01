import os
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import requests
from dotenv import load_dotenv
import random

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)

healthy_tips = [
    "Drink plenty of water every day.",
    "Get at least 7-8 hours of sleep each night.",
    "Exercise regularly to stay fit and healthy.",
    "Eat a balanced diet with plenty of fruits and vegetables.",
    "Take breaks and stretch during work hours.",
    "Practice mindfulness and meditation to reduce stress."
]

motivational_quotes = [
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Success is not the key to happiness. Happiness is the key to success. - Albert Schweitzer",
    "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
    "You are never too old to set another goal or to dream a new dream. - C.S. Lewis"
]

@bot.message_handler(commands=['start'])
def greet_user(message):
    first_name = message.from_user.first_name
    bot.reply_to(message, f"Hey {first_name}! 😊 How can I assist you today?")

    bot.send_message(message.chat.id, "⚡ *Quick Tips:* Use /commands to see commands!", parse_mode="Markdown")
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(
        InlineKeyboardButton("🐈 Meow", callback_data="meow"),
        InlineKeyboardButton("🆘 Help", callback_data="help"),
        InlineKeyboardButton("💡 Healthy Tip", callback_data="healthy_tip"),
        InlineKeyboardButton("💪 Motivation", callback_data="motivation"),
        InlineKeyboardButton("💬 Chat with Me", callback_data="call_me")
    )
    bot.send_message(message.chat.id, "Choose an option:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "meow")
def handle_meow(call):
    bot.send_message(call.message.chat.id, "🐈 Meow! 🐈")

@bot.callback_query_handler(func=lambda call: call.data == "help")
def handle_help(call):
    bot.send_message(call.message.chat.id, "I am here to help! Choose an option below:")

    # Create an inline keyboard with buttons for all commands
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(
        InlineKeyboardButton("💬 Chat with Me", callback_data="call_me"),
        InlineKeyboardButton("💡 Healthy Tip", callback_data="healthy_tip"),
        InlineKeyboardButton("💪 Motivation", callback_data="motivation"),
        InlineKeyboardButton("📜 Commands", callback_data="commands")
    )

    bot.send_message(call.message.chat.id, "Click a button below:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "call_me")
def handle_call_me(call):
    try:
        url = f"https://trigger-call-mqphszbpba-uc.a.run.app/?phone_number=%2B6591268057"
        response = requests.get(url)
        
        if response.status_code == 200:
            bot.send_message(call.message.chat.id, "Your call request has been queued.")
        else:
            bot.send_message(call.message.chat.id, "Failed to queue your call request. Please try again later.")
    except Exception as e:
        bot.send_message(call.message.chat.id, f"An error occurred: {e}")

@bot.callback_query_handler(func=lambda call: call.data == "healthy_tip")
def handle_healthy_tip(call):
    tip = random.choice(healthy_tips)
    bot.send_message(call.message.chat.id, f"💡 Healthy Tip: {tip}")

@bot.callback_query_handler(func=lambda call: call.data == "motivation")
def handle_motivation(call):
    quote = random.choice(motivational_quotes)
    bot.send_message(call.message.chat.id, f"💪 Motivation: {quote}")

@bot.callback_query_handler(func=lambda call: call.data == "commands")
def handle_commands(call):
    commands = [
        "/start - Start the bot",
        "/commands - List all commands",
        "/healthy_tip - Get a healthy lifestyle tip",
        "/motivation - Get a motivational quote"
    ]
    bot.send_message(call.message.chat.id, "\n".join(commands))

@bot.message_handler(commands=['commands'])
def send_commands(message):
    commands = [
        "/start - Start the bot",
        "/commands - List all commands",
        "/healthy_tip - Get a healthy lifestyle tip",
        "/motivation - Get a motivational quote"
    ]
    bot.send_message(message.chat.id, "\n".join(commands))

@bot.message_handler(commands=['healthy_tip'])
def send_healthy_tip(message):
    tip = random.choice(healthy_tips)
    bot.send_message(message.chat.id, f"💡 Healthy Tip: {tip}")

@bot.message_handler(commands=['motivation'])
def send_motivation(message):
    quote = random.choice(motivational_quotes)
    bot.send_message(message.chat.id, f"💪 Motivation: {quote}")

bot.infinity_polling()