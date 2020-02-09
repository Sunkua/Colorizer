import os
import random
def loadVideos(path):
    # load all .mp4-files from path into list
    os.chdir(path)
    video_list = [path +"/"+ f  for f in os.listdir(path) if f.endswith('.mp4')]
    return video_list

def saveToFile(video_list, filename):
    f = open(filename, "w+")
    for name in video_list:
        f.write(name + "\n")
    f.close()

def main():
    path = '/network-ceph/pgrundmann/youtube_processed'

    video_list = loadVideos(path)
    test_video_list = [video_list.pop(random.randrange(len(video_list))) for _ in range(500)]
    train_video_list = video_list

    random.shuffle(train_video_list)
    saveToFile(train_video_list, "/network-ceph/pgrundmann/youtube_processed/train_filenames.txt")
    saveToFile(test_video_list, "/network-ceph/pgrundmann/youtube_processed/test_filenames.txt")

if __name__ == "__main__":
    main()
