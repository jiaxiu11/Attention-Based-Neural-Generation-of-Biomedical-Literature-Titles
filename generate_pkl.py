# (heads, descs, keywords)
# heads is a list of headline stirngs
# descs a list of article strings in same order and length as heads
# keywords is None

import pickle as pkl
import pandas as pd
data = pd.read_csv("full_text.csv", header=None, names = ["title","abstract"])
data_dict = data.to_dict(orient='list')
print("done")
print(len(data_dict["title"]))
print(len(data_dict["abstract"]))

def remove_special_char(text):
    words = text.split(" ");
    alnum_words = ["".join(c for c in word if c.isalnum()) for word in words]
    return " ".join(alnum_words)

heads = [remove_special_char(t) for t in data_dict["title"]]
descs = [remove_special_char(d) for d in data_dict["abstract"]]
keywords = None

print(heads[0])
print(descs[0])

data = (heads, descs, keywords)

print ("writing to file")
fout = open("data.pkl", "wb")
pkl.dump(data, fout)
fout.close()
print ("complete")


