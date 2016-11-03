import os
import re
import argparse

keyword = 'Target mAP'

if __name__=='__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('log_file', help='Log file of trained model')

    args = args_parser.parse_args()

    log_file = open(args.log_file, 'r')
   
    mAP_to_model = dict()
    for line in log_file:
        if keyword in line:
            start_idx = line.find(keyword)
            line = line[start_idx:]
            mAP = re.findall("\d+\.\d+", line)[0] # Will have only one float value
            next_line =  next(log_file)
            for item in next_line.split():
                if "caffemodel" in item:
                    mAP_to_model[mAP] = os.path.basename(item)

    max_mAP = max(mAP_to_model.keys(), key=float)
    print '{} has largest mAP {}'.format(mAP_to_model[max_mAP], max_mAP)
