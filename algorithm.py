import json

# Open and read the JSON file
data = json.load(open('rules and responses.json', 'r'))

input_topic = input("input: ")
input_classifcation = input("classification: ")

if input_classifcation == "word":
    print(f"    I'm not sure, what kind of things do you think {input_topic} is?,")
    print(f"    I don't know, could you explain the concept of {input_topic}?,")
elif input_classifcation == "a word":
    print(f"    I'm not sure, what do you think a {input_topic} is?,")
    print(f"    I don't know, could you explain the concept of a {input_topic}?,")
elif input_classifcation == "are word":
    print(f"    I'm not sure, what kind of things do you think are {input_topic}?,")
    print(f"    I don't know, could you explain what should be {input_topic}?,")
