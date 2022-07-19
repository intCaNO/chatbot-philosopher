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

  아래의 두 개의 코드는 각각 nltk.utils.py, 그리고 model.py이다. 이 파이썬 코드들은 사용자의 입력을 예측하고 분류하는데 필요한 nltk의 punkt 토크나이저(tokenizer) 시스템과 인공지능 커뮤니케이션 시스템의 기반이 되어줄 뉴런 시스템인 NeuralNet 모듈을 구현한 것이다. nltk.utils.py는 파이썬의 외장 모듈인 nltk와 numpy를 사용하여 위에서 기술한 데이터베이스인 intents.json 파일에서 인풋 데이터를 가져와 단어들을 어근 단위로 청크화 시켜 일종의 스템(stem) 형태의 토큰을 생성한다. 이 때, 중복된 스템을 걸러내기 위한 코드를 추가해 데이터의 중복을 방지하였다. 그리고 model.py는 전체적인 시스템이 돌아가기 위한 가장 기본적인 베이스인 NeuralNet을 불러오고 인공지능 커뮤니케이션 모델에 공간을 할당하는 작업을 수행한다.

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
  
```python
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
```

다음은 전체 프로그램의 핵심 코드 중 하나인 인공지능에게 데이터베이스를 학습시키는 프로그램이다. 상술했던 프로그램들은 전체적인 구현을 위해서 필요한 사전 작업 같은 것이였다면, 본 train.py르 포함한 후술할 코드 브렌치들은 실질적으로 챗봇이 돌아가기 위한 작업을 수행하는 프로그램 코드들이다. 한편 train.py 만큼은 전반적인 시스템의 핵심이 되는 코드이기에 조금 더 세부적으로, 코드별로 뜯어서 진행과정과 프로그램의 논리 흐름을 설명하고자 한다.

```python
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
```

여기까지의 구문은 train.py를 구동하기 위해 전에 프로그래밍 해놓은 베이스 파일들을 현재 프로그램 안에서 사용할 수 있게 import 명령어로 불러오는 작업을 수행한다. 이 때, array들을 처리할 수 있게 해주는 모듈인 numpy, 챗봇이 대답을 하는 과정에서 데이터베이스에서 랜덤한 하나의 대답을 불러오는데 사용될 random 모듈, 데이터베이스와의 연결을 위한 json 모듈, 그리고 인공지능의 연산시스템의 기반이 되는 각종 pytorch 모듈과 NeuralNet을 불러왔다. 코드 줄에서 'with open('intents.json','r') as f: 와 따라오는 블록 안의 intents = json.load(f)는 위에서 제작했던 데이터베이스에서 직접 정보를 가져오는 역할을 수행한다.

```python
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
```

이 코드들은 입력받은 json 파일의 데이터베이스에서 스템(stem)을 생성하는 역할을 담당한다. 이는 stem을 생성하기 위한 작업을 수행하는 코드들(기호나 특수문자들을 없애는 작업, 반복되는 stem들을 제거하고 정렬하여 보여주는 작업)을 포함한다.

```python
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
```
다음 코드들은 훈련 데이터를 만드는 작업을 수행한다. X열과 Y열의 두가지의 집합체를 이용하고 이들은 앞에서 import 해온 numpy를 이용해 계산하고 학습되어진다. 이떄, X열에는 json 데이터베이스로부터 만들어낸 pattern 문장들이, Y열에는 후술할 Pytorch 프로그램을 이용한 훈련 과정에서 라벨링 된 훈련 로스율 데이터가 변수로 저장된다.

```python
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)
```

위의 코드들은 인공지능의 학습 과정을 위해 필요한 파라미터들이다.

```python
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

이 부분은 학습 과정에서 생기는 오류 격차를 수치화시키고, 줄여나가기 위한 변수를 설정하는 단계이다. 코드를 보면 criterion에 학습 과정에서 생기는 엔트로피 로스 값인 CrossEntropyLoss가 담겨 있다. 또한 기본적인 데이터 트레이닝을 위한 함수들(코드에서 def로 정의된 사용자 정의 함수들)과 변수들(dataset, train_loader, device, model, criterion, optimizer)이 저장되어 있다.

```python
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        

        outputs = model(words)
        loss = criterion(outputs, labels)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
```

위의 코드는 실제로 학습 프로그램이 동작하는 부분이다. 이 때 코드는 오차율 내에서 손실율이 0에 가깝게 수렴할 때까지 반복하며 계속해서 학습을 진행해 손실율을 줄여나간다.

```python
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
```

그리고 특정 손실율에 도달하여 학습이 종료되면 프로그램을 학습 데이터를 data.pth라는 모델 파일에 저장하고 학습을 마친다.
여기까지 코드를 진행했다면 전체 프로그램에는 data.pth라는 훈련 데이터가 존재하게 된다. 즉, 이제 실제로 챗봇을 구동할 준비가 완료된 것이다. 그리고 다음 코드는 실제로 인공지능 커뮤니케이션 시스템을 돌아가게끔 하는 챗봇의 본체 코드인 chat.py 이다.
```python
import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Supinoza"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "It's not philosophical..."


if __name__ == "__main__":
    print("Hi! It's Supinoza, your A.I. philosopher.")
    print("I'm impressed with German Idealism or Deutscher Idealismus.(type 'no' for exit)")
    print("Let's have any conversation!")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
```

chat.py는 사용자에게 대화 인풋 값을 받아 변수에 저장하고, 변수에 저장한 사용자의 말을 토큰 단위로 분석해 자신이 가지고 있는 기존의 베이스 데이터인 data.pth와 비교하여 적절한 답변을 내놓는다. 다음
