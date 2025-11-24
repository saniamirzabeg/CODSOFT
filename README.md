CODSOFT Task 1
Simple Rule-Based Chatbot

print("Chatbot: Hello! I am a simple rule-based chatbot.")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").lower()

    if "hello" in user_input or "hi" in user_input:
        print("Chatbot: Hello! How can I help you today?")
    
    elif "your name" in user_input:
        print("Chatbot: I am a rule-based chatbot created using Python.")
    
    elif "how are you" in user_input:
        print("Chatbot: I'm just a program, but I'm working perfectly!")
    
    elif "weather" in user_input:
        print("Chatbot: I cannot check weather, but I hope it's nice where you are!")
    
    elif "bye" in user_input:
        print("Chatbot: Goodbye! Have a great day!")
        break

    else:
        print("Chatbot: Sorry, I don't understand that. Try asking something else.")
