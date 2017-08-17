'''
Script for preparing lmdb files for pretraining, and 
creates softlinks to datasets under data/bgfr_<corpora-name>/<experiment-name>/
'''
import os
import shutil
import subprocess

# Caffe LMDB root on data-store
lmdb_rootdir = '/home/data-store/caffe_lmdb'
# Root dir of corpora
corpora_rootdir = '/home/data-store/corpora'
# Current corpora name. Change this accordingly
corpora = '2016-02-10_cnnG_four-corpora-887'
# Directory where softlinks of created lmdb files has to be placed
softlink_dir = os.path.join('data/bgfr_four_corpora', corpora)
# Input dimensions
height = 84
width = 84

# Remove any old links if exists
if os.path.isdir(softlink_dir):
    shutil.rmtree(softlink_dir)
    
os.makedirs(softlink_dir)

for fold in ['pretrain.train', 'pretrain.val']:
    # Indices file
    idx_file = os.path.join(lmdb_rootdir, corpora, 'indices', '{}.list'.format(fold))
    # LMDB destination
    dest_db = os.path.join(lmdb_rootdir, corpora, 'data', '{}_lmdb'.format(fold))
    # Create or skip LMDB file creation
    if os.path.exists(dest_db):
        print '{} exists. Skipping lmdb files creation'.format(dest_db)
    else:
        cmd = 'build/tools/convert_imageset --check_size=true --gray=true \
               --resize_height={} --resize_width={} {} {} {}'.format(height,
                                  width, corpora_rootdir, idx_file, dest_db)
        subprocess.call(cmd, shell=True)
    # Create softlink
    cmd = 'ln -s {} {}'.format(dest_db, os.path.join(softlink_dir, 
                                                '{}_lmdb'.format(fold)))
    subprocess.call(cmd, shell=True)
