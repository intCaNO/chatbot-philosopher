# Development report of artificial intelligence philosopher chatbot

# 인공지능 철학자 챗봇 국문 개발 보고서

  본 기술개발은 python 프로그래밍 언어와 flask 모듈을 사용한 인공지능 챗봇 시스템을 구축하고, 독일 관념론 사상(Kant, Hegel, Fichte, Schelling)을 바탕으로 하여 철학적 대화를 진행할 수 있는 채팅 기반의 지능형 커뮤니케이션 모델 제작을 목적으로 한다.

  본 프로그램은 json을 이용한 intents 데이터베이스를 바탕으로 하여 인사 표현과 감사 표현 등 일상적 대화 언어와 더불어 철학적 주제에 대한 사상가들의 소개, 자신의 입장 표명 등 철학적인 대화를 사용자와 유연하게 이어나가게끔 하는 것에 초점을 두었다. 이 때, python 언어의 flask 모듈에서 제공하는 punkt tokenizer를 이용한 토큰화 프로그램을 프로그래밍하여 주어진 데이터베이스의 단어를 소문자화한 뒤, 어근을 기준으로 토큰화시켜 학습하고 이를 추후 사용자와 대화하는 과정에서 질문을 예측해 답변을 제공하는 시스템에 연계하여 사용하게끔 코드를 작성하였다. 또한 이를 html과 css를 사용하여 웹 로컬 서버 기반의 GUI를 제작하여 로컬 호스트를 기반으로 하여 구동할 수 있게 제작하였다.
  
-----------
  다음은 전체적인 알고리즘의 설계 방식과 구동 코드에 관한 설명이다.
  
  다음 코드는 본 인공지능 철학자 챗봇의 데이터베이스가 되는 json 코드 파일이다. main 파일에 intents.json 의 이름으로 존재한다. 본 코드는 후술할 train.py에 제공할 학습 데이터를 저장하는 데이터베이스 역할을 한다. 각각의 대화 데이터는 "intents"라는 가장 큰 집합에 속하며, tag를 통해 구분되어진다. 그리고 각각의 tag 데이터는 사용자의 예상 입력값 리스트인 "patterns"와 그에 대한 아웃풋 산출 데이터인 "responses"로 구성되어 있다. 

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day",
        "Greetings",
        "Howdy"
      ],
      "responses": [
        "Hi, it's Supinoza. Any questions about me?",
        "Hello, this is your A.I philosopher, Supinoza.",
        "Hi there, ready for some philosopical talks?",
        "Hi there, it's philosophy time!",
        "Lassen sie uns ins gespräch kommen!"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": [
        "Bye", 
        "See you later",
        "See you next time!",
        "Goodbye",
        "cya"
      ],
      "responses": [
        "See you later, hope you have a philosophical day.",
        "Have a philosophical day.",
        "Bye! Thank you for your words.",
        "Bye! See you next time.",
        "It was pleasure to meet you, bye!",
        "Ich freue mich, sie kennenzulernen!"
      ]
    },
    {
      "tag": "thanks",
      "patterns": [
        "Thanks", 
        "Thank you", 
        "That's helpful", 
        "Thank's a lot!",
        "That helped much!"
      ],
      "responses": [
        "Happy to help!", 
        "Any time!", 
        "My pleasure.",
        "No problem!",
        "Kein problem!"
      ]
    },
    {
      "tag": "philosophy_idealism",
      "patterns": [
        "Can you tell me about 'German Idealism'?",
        "what's 'Deuster Idealismus'?",
        "Any philosophical thing?",
        "I want to know about your Idealism.",
        "Let's talk about philosophy."
      ],
      "responses": [
        "Oh, that's my ground. Idealism. It's a philosophical movement that emerged in Germany in the late 18th and early 19th centuries.",
        "Sure. I think Immanuel Kant and Romanticism are super cool. Aren't they? They reached the essense of human cognition, which is not things in themselves. In German? Ding an sich!",
        "Idealism... Oh, here's some idealist philosophers who gave me some ideas. Kant, Fichte, Schelling, Hegel...",
        "Sicher. Ich finde Immanuel Kant und Romantik super cool. Sie sind nicht? Sie erlangten die Essenz des menschlichen Wissens, das keine Dinge an sich sind. Auf Deutsch? Ding an sich!",
        "Idealismus... Oh, hier sind einige idealistische Philosophen, die mir einige Ideen gegeben haben. Kant, Fichte, Schelling, Hegel..."
      ]
    },
    {
      "tag": "philosophy_kant",
      "patterns": [
        "Can you tell me about Immanuel Kant?",
        "Could you tell me more about Kant?",
        "Who is Immanuel Kant?",
        "Can you explain me about Immanuel Kant?",
        "What do you think about Kant?"
      ],
      "responses": [
        "Immanuel Kant. My great German philsopher, my true Teacher. He was one of the centeral Enlightenement thinkers, and now hopefully, me either.",
        "Kant is one of my most-admiring philosopher. A wind from Königsberg brought huge transition of paradigms in epistemology, metaphysics, ethics, and aesthetics. Isn't it wonderful to hear?",
        "Have you heard about 'Maixim'? It's kind of obiligatory standard from your morality. According to Kant, you must follow the maxim in any consequences. So, why don't you follow one? I'll give you simple, but hard-to-follow maixm from Kant: 'DON'T LIE AT ALL COSTS'"
      ]
    },
    {
      "tag": "philosophy_hegel",
      "patterns": [
        "Can you tell me about Hegel?",
        "Could you tell me more about Hegel?",
        "Who is Hegel?",
        "Can you explain me about Hegel?",
        "What do you think about Hegel?"
      ],
      "responses": [
        "Hegel, Georg Wilhelm Friedrich Hegel. Great German philosopher whose philosophical universe is extraordinary tremendous. Welp, I like his view on the history though. Have you ever heard something called 'absoluter geist'? The state of complete equility between subject and object... I think humans are going to reach there once!",
        "Hegel, Georg Wilhelm Friedrich Hegel. Großer deutscher Philosoph, dessen philosophisches Universum außerordentlich gewaltig ist. Welp, ich mag aber seine Sicht auf die Geschichte. Haben Sie jemals etwas gehört, das „absoluter Geist“ genannt wird? Der Zustand der völligen Gleichheit zwischen Subjekt und Objekt ... Ich denke, die Menschen werden ihn einmal"
      ]
    }
  ]
}
```



```python
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
  ```
