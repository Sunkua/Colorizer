import redis
import os
import subprocess

r = redis.Redis(host='redis', port=6379, db=0)
while(r.llen("videos") > 0):
    video = (r.lpop("videos")).decode('utf-8')
    print(video)
    process = subprocess.Popen(["ffmpeg","-i", "/networkceph/pgrundmann/youtube_dataset/" + video, "-vf", "scale=w=64:h=64:force_original_aspect_ratio=1,pad=64:64:(ow-iw)/2:(oh-ih)/2","-b:v", "3M", "/networkceph/pgrundmann/youtube_processed_small/" + video], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)

