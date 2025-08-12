import json


# process the raw json data and convert it to usable text files
def raw_to_processed(raw_data, output_file):
    # Loop over data and write to input file
    data_looper = 0
    # only half the length since it will be using two items at a time
    while data_looper < int(len(raw_data)/2):
        # Get the question, every even number are the strings
        question = raw_data[data_looper*2]
        # Loop over the answers, every odd number are the lists
        for answer in raw_data[data_looper*2+1]:
            # Add the same amount of newline characters
            #  as block_size in the training script
            # The reason why this is done is because
            #  the ai will incorrectly learn to try
            #  to predict more questions after answering
            #  this not only is not something it should learn
            #  but could cause issues and confusion for users
            #  if it seemingly starts getting new weird inputs
            #  that weren't typed in
            for i in range(128):
                output_file.write('\n')
            # Print question then answer
            output_file.write(f'question: "{question}"\n')
            output_file.write(f'answer: "{answer}"\n')

        data_looper += 1


# Open and read the raw training and testing data from the JSON files
raw_training_data = json.load(open('questions.json', 'r'))
raw_testing_data = json.load(open('testing.json', 'r'))
# Output to these file for training and validating the AI
input_for_ai = open('input.txt', 'w')
validation_for_ai = open('validation.txt', 'w')

raw_to_processed(raw_training_data, input_for_ai)
raw_to_processed(raw_testing_data, validation_for_ai)
