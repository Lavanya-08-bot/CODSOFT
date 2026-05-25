import random
from datetime import datetime


class SimpleChatbot:

    def __init__(self):
        self.conversation_history = []
        self.responses = self.initialize_responses()

    def initialize_responses(self):

        return {

            "greeting": [
                "Hello! How can I help you today?",
                "Hi there!",
                "Hey! Nice to meet you.",
                "Greetings!"
            ],

            "farewell": [
                "Goodbye! Have a nice day!",
                "See you later!",
                "Take care!",
                "Bye! 👋"
            ],

            "thankyou": [
                "You're welcome!",
                "Happy to help!",
                "No problem!",
                "Anytime 😊"
            ],

            "name_question": [
                "I'm ChatBot.",
                "You can call me Assistant.",
                "My name is Simple ChatBot."
            ],

            "age_question": [
                "I don't have an age like humans.",
                "I'm just a computer program!",
                "Age doesn't apply to bots 😄"
            ],

            "weather_question": [
                "I cannot provide live weather updates.",
                "Please check a weather app for forecasts.",
                "Try Google Weather or AccuWeather."
            ],

            "time_question": [
                f"Current time is {datetime.now().strftime('%I:%M %p')}",
                f"The time now is {datetime.now().strftime('%H:%M')}"
            ],

            "date_question": [
                f"Today's date is {datetime.now().strftime('%d-%m-%Y')}",
                f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
            ],

            "joke": [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why was the computer cold? It left its Windows open!",
                "Why did the programmer quit? Because he didn't get arrays!"
            ],

            "help_request": [
                "You can ask me about time, date, jokes, greetings and more.",
                "Try typing: hello, joke, time, who are you, bye"
            ],

            "how_are_you": [
                "I'm doing great!",
                "I'm fine. Thanks for asking!",
                "All systems are working perfectly ⚡"
            ],

            "sad_emotion": [
                "I'm sorry you're feeling sad.",
                "Things will get better soon.",
                "I'm here to chat with you."
            ],

            "happy_emotion": [
                "That's wonderful 😊",
                "Glad to hear that!",
                "Keep smiling!"
            ],

            "default": [
                "I don't understand that.",
                "Could you rephrase your sentence?",
                "I'm still learning."
            ]
        }

    def preprocess_input(self, text):

        text = text.lower().strip()

        for char in ['.', ',', '!', '?']:
            text = text.replace(char, '')

        return text

    def match_intent(self, text):

        text = self.preprocess_input(text)

        patterns = {

            "greeting": [
                "hello", "hi", "hey",
                "good morning", "good evening"
            ],

            "farewell": [
                "bye", "goodbye", "exit",
                "quit", "see you"
            ],

            "thankyou": [
                "thanks", "thank you", "thx"
            ],

            "name_question": [
                "who are you",
                "your name",
                "what is your name"
            ],

            "age_question": [
                "how old are you",
                "your age",
                "age"
            ],

            "weather_question": [
                "weather", "rain",
                "forecast", "temperature"
            ],

            "time_question": [
                "time", "clock",
                "current time"
            ],

            "date_question": [
                "date", "today",
                "day", "month", "year"
            ],

            "joke": [
                "joke", "funny",
                "make me laugh"
            ],

            "help_request": [
                "help", "support",
                "guide", "commands"
            ],

            "how_are_you": [
                "how are you",
                "are you fine"
            ],

            "sad_emotion": [
                "sad", "upset",
                "depressed", "problem"
            ],

            "happy_emotion": [
                "happy", "awesome",
                "great", "excited"
            ]
        }

        # Check every pattern
        for intent, keywords in patterns.items():

            for keyword in keywords:

                if keyword in text:
                    return intent

        return "default"

    def generate_response(self, user_input):

        intent = self.match_intent(user_input)

        response = random.choice(self.responses[intent])

        self.conversation_history.append({
            "user": user_input,
            "bot": response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return response

    def display_stats(self):

        print("\n===== CHAT STATS =====")
        print("Total Messages:", len(self.conversation_history))
        print("======================")

    def clear_history(self):

        self.conversation_history.clear()
        print("Conversation history cleared!")

    def save_conversation(self):

        with open("conversation_log.txt", "w", encoding="utf-8") as file:

            for item in self.conversation_history:

                file.write(f"User: {item['user']}\n")
                file.write(f"Bot : {item['bot']}\n")
                file.write("-" * 40 + "\n")

        print("Conversation saved successfully!")



def start_chatbot():

    print("=" * 50)
    print("🤖 SIMPLE RULE BASED CHATBOT")
    print("=" * 50)

    bot = SimpleChatbot()

    while True:

        user_input = input("You: ")

        processed = bot.preprocess_input(user_input)

        if processed in ["bye", "exit", "quit", "goodbye"]:

            print("Bot: Goodbye! 👋")
            bot.save_conversation()
            break

        elif processed == "stats":

            bot.display_stats()
            continue

        elif processed == "clear":

            bot.clear_history()
            continue

        response = bot.generate_response(user_input)

        print("Bot:", response)
        print()


if __name__ == "__main__":

    try:
        start_chatbot()

    except KeyboardInterrupt:
        print("\nChatbot stopped.")

    except Exception as e:
        print("Error:", e)