# This will likely be fundamentally changed in a later commit

import json

# Open and read the raw training data from the JSON file
raw_training_data = json.load(open('questions.json', 'r'))
# Output to this file for training the AI
input_for_ai = open('input.txt', 'w')

# Loop over data and write to input file
data_looper = 0
while data_looper < int(len(raw_training_data)/2):
    # Get the question
    question = raw_training_data[data_looper*2]
    # Loop over the answers
    for answer in raw_training_data[data_looper*2+1]:
        # Add the same amount of newline characters
        #  as block_size in the training script
        # The reason why this is done is because
        #  the ai will learn to try to predict more questions after answering
        #  this not only is not something it should learn
        #  but could cause issues and confusion for users
        for i in range(128):
            input_for_ai.write('\n')
        # Print question then answer
        input_for_ai.write(f'question: "{question}"\n')
        input_for_ai.write(f'answer: "{answer}"\n')

    data_looper += 1
