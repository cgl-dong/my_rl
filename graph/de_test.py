import collections

buffer = collections.deque(maxlen=10)
for i in range(100):
    buffer.append(i)
    print(buffer)