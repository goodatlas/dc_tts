formatted transcript and vocab for jamo-level training

- fixed transcript text from NFD to NFC using `unicodedata.normalize()`
- used jamotools.split_syllables() for positionless jamo
- lower-case
- replaced initial ㅇ with null symbol ⅇ
- replaced spaces with ▁
- formatted `transcript.csv` in LJSpeech format (filename, <sent>, sent data, duration)
- however must SPLIT ON SPACE! not just `list(sent)`

```
# sent2jamo
# replace initial ㅇ with null
def sent2jamo(sent, null='ⅇ'):
    sent = sent.lower()
    seq = []
    for c in list(sent):
        if re.match(r'[가-힣]', c):
            jamos = list(jamotools.split_syllables(c)) # use this for positionless
            if jamos[0] == 'ㅇ': # the 'positionless' nieung
                jamos = [null] + jamos[1:]
            # print(jamos)
            seq += jamos
        else:
            if c == ' ':
                c = '▁'
            seq.append(c)
    return ' '.join(seq)
```

test sents (orig)

```
동해물과 백두산이 마르고 닳도록
하느님이 보우하사 우리나라 만세.
나는 아틀라스 랩스의 제로스 티티애스입니다.
만나서 반갑습니다.
앞으로 잘 부탁드립니다.
혹시 딥러닝 아세요?
우리 회사가 인공지능 회사 입니다.
영어 할 수 있어요?
한국말 잘 못해요.
천천히 말씀해 주세요.
보기 좋은 떡이 먹기도 좋습니다.
로마는 하루아침에 이루어진 것이 아니다.
과부 설움은 홀아비가 안다.
낮말은 새가 듣고 밤말은 쥐가 듣는다.
눈에서 멀어지면, 마음에서도 멀어진다.
장님이 코끼리 만지는 격이다.
다 된 밥에 재 뿌리지 마라.
될성부른 나무는 떡잎부터 알아본다.
원숭이도 나무에서 떨어질 때가 있습니다.
가는 말이 고와야오는 말이 곱다.

```