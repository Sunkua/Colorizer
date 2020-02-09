import redis
import glob
import os
import time


r = redis.Redis(host='redis', port=6379, db=0)
while r.llen("videos") > 0:
    print(str(r.llen("videos")), end="\r")
    time.sleep(2)