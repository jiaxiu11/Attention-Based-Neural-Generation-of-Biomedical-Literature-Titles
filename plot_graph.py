import matplotlib.pyplot as plt

loss=[]
with open("res/train3.out", 'r') as fp:
    for l in fp:
        tokens = l.strip().split()
        if len(tokens)==0: continue
        iterations = tokens[0].split("/")
        if len(iterations) != 2: continue
        if iterations[1]=="1875": loss.append(float(tokens[len(tokens)-1]))
print(loss)
plt.plot(loss, color="black")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()