import tensorflow as tf
import os, pathlib, shutil, random

# get the data
#   wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#   tar -xf aclImdb_v1.tar.gz
#   rm -r aclImdb/train/unsup
