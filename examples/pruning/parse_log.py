import os
import re
import argparse

target = 'Target mAP'
eval = 'detection_eval'

if __name__=='__main__':
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('log_file', help='Log file of trained model')
    args_parser.add_argument('--target', help='Search for model with highest target mAP', action='store_true')
    args_parser.add_argument('--eval', help='Search for model with highest eval mAP', action='store_true')

    args = args_parser.parse_args()

    log_file = open(args.log_file, 'r')
   
    mAP_to_model = dict()
    for line in log_file:
        if args.target:
            if target in line:
                start_idx = line.find(target)
                line = line[start_idx:]
                mAP = re.findall("\d+\.\d+", line)[0] # Will have only one float value
                next_line =  next(log_file)
                for item in next_line.split():
                    if "caffemodel" in item:
                        mAP_to_model[mAP] = os.path.basename(item)
        elif args.eval:
            if eval in line:
                start_idx = line.find(eval)
                line = line[start_idx:]
                mAP = re.findall("\d+\.\d+", line) # Will have only one float value
                if len(mAP) == 1:
                    mAP = mAP[0]
                    next_line =  next(log_file)
                    start_idx = next_line.find('Iteration')
                    next_line = next_line[start_idx:]
                    next_line =  next_line.split()
                    if len(next_line) > 1:
                        itr = next_line[1]
                        mAP_to_model[mAP] = os.path.basename(itr)

 
    max_mAP = max(mAP_to_model.keys(), key=float)
    print '{} has largest mAP {}'.format(mAP_to_model[max_mAP], max_mAP)
