from helpbot import HelpBot

def main():
    """
    Main program function
    """
    bot = HelpBot()
    while True:
        statement = input("You: ")
        exit_keywords = ["exit", "bye", "quit", "stop", "end"]
        if statement.lower() in exit_keywords:
            print("Goodbye!")
            break
        response = bot.respond_to_user(statement)
        print("HelpBot:", response)

if __name__ == "__main__":
    main()
