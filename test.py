import time

for i in range(100):
    for j in range(10):
        if j > 5:
            break
        print("%d-%d\r" % (i, j), end='')
        time.sleep(0.2)
