import subprocess
import os

rand_seeds = range(0,100)

for seed in rand_seeds:
    print("Running seed: " + str(seed))
    subprocess.call(['python', '/home/fhh/ember-master/MalConv2-main/testtrain.py',
                      "--randseeds", str(seed),
                      "--epochs","5",
                "/home/fhh/ember-master/MalConv2-main/dataset/malconv2_mal_train_10000",
                "/home/fhh/ember-master/MalConv2-main/dataset/malconv2_ben_train_10000",
                "/home/fhh/ember-master/MalConv2-main/dataset/malconv2_mal_test_2000",
                "/home/fhh/ember-master/MalConv2-main/dataset/malconv2_ben_test_2000",])