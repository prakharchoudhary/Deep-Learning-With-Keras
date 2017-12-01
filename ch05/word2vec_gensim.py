from gensim.models import word2vec
import logging
import os
import io


class Text8Sentences(object):

    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen

    def __iter__(self):
        with io.open(os.path.join(DATA_DIR, "text8"), "r") as ftext:
            text = ftext.read().split(" ")
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

DATA_DIR = "./data/"
sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
model = word2vec.Word2Vec(sentences, size=300, min_count=30)

model.init_sims(replace=True)
model.save("word2vec_gensim.bin")
model = word2vec.Word2Vec.load("word2vec_gensim.bin")

print("""model.most_similar("woman")""")
print(model.most_similar("woman"))

print(
    """model.most_similar(positive=["woman", "king"], negative=["man"], topn=10)""")
print(model.most_similar(positive=['woman', 'king'],
                         negative=['man'], topn=10))

print("""model.similarity("girl", "woman")""")
print(model.similarity("girl", "woman"))
print("""model.similarity("girl", "man")""")
print(model.similarity("girl", "man"))
print("""model.similarity("girl", "car")""")
print(model.similarity("girl", "car"))
print("""model.similarity("bus", "car")""")
print(model.similarity("bus", "car"))
