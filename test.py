import time

for i in range(100):
    print("%i\r" % i, end='')
    time.sleep(0.5)
