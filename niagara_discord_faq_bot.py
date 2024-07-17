""" This is a very simple skeleton for a FAQ bot, based on the Assignment 0.
It creates discord FAQ bot that can answer 20 questions related to
Niagara Fals using basic string matching.
modify By
Kruti Patel(000857563)  27/03/2023
Sam Scott, Mohawk College, May 2021
"""
import discord
import openai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
openai.api_key = "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"
clf = load('classifier.joblib') #load pickled classifier for sentiment analysis
vectorizer = load('vectorizer.joblib') #load the pickled vectorizer for sntiment analysis
topic_clf = load('classifier_topic.joblib') #load pickled classifier for topic-off-topic analysis
topic_vectorizer=load('vectorizer_topic.joblib') #load pickled vectorizer for topic-off-topic analysis
import textwrap

def file_input(filename):

    """Loads each line of the file into a list and returns it."""
    lines = []
    with open(filename) as file: # opens the file and assigns it to a variable
        for line in file:
            lines.append(line.strip()) # the strip method removes extra whitespace
    return lines


#nnnkmm
def load_FAQ_data():
    """This method returns a list of questions and answers. The
    lists are parallel, meaning that intent n pairs with response n."""
    docs = file_input("questions.txt")
    answers = file_input("answers.txt")

    chunk_size = 3 # no of version of questions
    sublists=[]
    for i in range(0, len(docs), chunk_size): # divide the list in to sublist of size 3
        sublists.append(docs[i:i+chunk_size])
    #print(sublists)
    return docs, answers ,sublists

intents,responses,sub_intents = load_FAQ_data()

def understand(utterance):

    """This method processes an utterance to determine which intent it
    matches. The index of the intent is returned, or -1 if no intent
    is found."""

    global intents # declare that we will use a global variable
    global sub_intents

    vectorizer = CountVectorizer(token_pattern=r"(?u)(\b\w+\b|[?.!])",stop_words='english',ngram_range=(1,2),max_df=0.3)# using paramers for best performance
    vectors = vectorizer.fit_transform(intents)
    questions=vectors.toarray()
    #print(questions[78])

    new_vector = vectorizer.transform([utterance])
    similarities = cosine_similarity(new_vector, vectors)[0]>0.70 # checking for optimum value.
    #print(similarities)
    try:
        max = -1
        for i in range(len(similarities)): # search for best similar question
            if similarities[i] > similarities[max]:
                max = i # assignning the index of best similar question
        #print(max)
        return max #return the index of the best match question.
    except ValueError:
        return -1


def generate(intent,utterance):

    """This function returns an appropriate response given a user's
    intent."""

    vector_utterance = vectorizer.transform([utterance])
    prediction = clf.predict(vector_utterance)

    topic_utterance = topic_vectorizer.transform([utterance])
    topic_prediction = topic_clf.predict(topic_utterance)
    global responses # declare that we will use a global variable

    if intent == -1:

        if (prediction[0] == 1 and topic_prediction[0] == 1): # if the sentiment is positive and it's on topic
            response = tune_using_gpt3(utterance, 1, 1)  # use GPT-3 to generate response
        if (prediction[0] == 1 and topic_prediction[0] == 0): # if the sentiment is positive and but it' on off topic
            response = tune_using_gpt3(utterance, 1, 0)  # use GPT-3 to generate response

        if (prediction[0] == 0 and topic_prediction[0] == 1): # if the sentiment is negative but it's on topic
            response = tune_using_gpt3(utterance, 0, 1)  # use GPT-3 to generate response
        if (prediction[0] == 0 and topic_prediction[0] == 0): # if the sentiment is negative and on off topic
            response = tune_using_gpt3(utterance, 0, 0)  # use GPT-3 to generate response

        return response #return response using gpt-3


    value_to_find=intents[intent]

    sub_intents_index = -1

    for i in range(len(sub_intents)): # checking the inedex of the best match question in sublist to get the answer
        if value_to_find in sub_intents[i]:
            sub_intents_index = i
            #print(sub_intents_index)
            break

    return responses[sub_intents_index] #return response related to intent

def tune_using_gpt3(utterance,prediction,topic):

    """   This function returns the appropriate respose using GPT-3 davinci-003 model.
    """
    test = " "

    if (prediction == 0 and topic == 0):  # sentiment is negative and it's off topic
        prompt += "Reply to this, which was said in a negative tone, while talking about other things:"
    elif (prediction == 1 and topic == 1): # if the sentiment is positive but its on topic
        prompt += "Reply to this, which was said in a absolute positive tone,while making the conversation about Niagara Falls:"
    elif (prediction == 1 and topic == 0): # sentiment is positive and it's off topic
        prompt += "Reply to this, which was said in a positive tone, while making the conversation about Niagara Fall:"
    elif (prediction == 0 and topic == 1): # i sentiment is negative but its on topic
        prompt += "Reply to this, which was said in a negative tone and while talking about topic"
    else:
        prompt+="Reply with positive tone and try to divert user to stay on topic"
    prompt+=f"{utterance}.answer"
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=300, temperature=0.7)
    return response.choices[0].text.strip() #returns the gpt3 generated responce.


class MyClient(discord.Client):

    def __init__(self):

        """ this method is a special method that is called when an instance of the class is created.  """

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self):
        """ This method is called when the bot is connected and ready to receive commands. """

        print('Logged on as', self.user)

    async def on_message(self, message):
        """ This method is called when a message is sent in the server.It takes in a message object that contains the content of the message, sender information, and other information related to the message. """
        self.prefix = "!"
        # don't respond to ourselves
        if message.author == self.user or not message.content.startswith(self.prefix):
            return
        # Get the command and arguments
        command = message.content[len(self.prefix):].split()[0]
        args = message.content[len(self.prefix) + len(command):].strip()

        # get the utterance and generate the response

        utterance = message.content
        list=""
        intent = understand(utterance)
        response = generate(intent,utterance)

        # start conversesion when user type 'hello'

        if message.content.startswith('!hello'):
            await message.channel.send('Hello! I have some information about Niagara Falls \n You can ask me question and I will try to give you ansawers.\n When you are done talking, just say "goodbye"')

        #when user type goodbye. it exit the program.

        elif message.content.startswith('!goodbye'):

            await message.channel.send('Nice talking to you!')

        else:
        # send the response
            await message.channel.send(response)

client = MyClient()
with open("bot_token.txt") as file:
    token = file.read()
client.run(token)

""" Resources used for question answers :
https://www.geeksforgeeks.org/print-lists-in-python-4-different-ways/
https://www.niagarafallstourism.com/
https://beta.openai.com/playground/
https://chat.openai.com/chat
https://www.niagarafallslive.com/
https://en.wikipedia.org/wiki/Niagara_Falls
https://gpt3demo.com/apps/how-to-build-a-gpt-3-chatbot-with-python
https://cobusgreyling.me/how-to-create-a-gpt-3-chatbot-in-12-lines-of-code/


"""