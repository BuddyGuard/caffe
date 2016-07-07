#!/bin/bash
# evaluate the results in path/to/corpora/bgobjects/SRC/results
# demonstrates how the buddy_evaldet.py can be directly called from corpora

## possible command argumentss :
# 0 1 0 # voc data evaluation for cat, dog and person
# 1 0 person # single class person detector evaluation for vatic bgobjects set
# 1 0 dog
# 1 0 cat
# 1 0 1  # cat, dog and person for vatic bgobjects set

if [ -z $3 ]; then
    echo -e "Usage: $0 <bgobjects>  <denormalize>  <reduced> <min_overlap>\n bgobjects: 1, denormalize: 0,\
    reduced = 0 if using bgobjects set with the ref yolo-tiny model,\n reduced = 1 if using a reduced model\
    min_overlap=0.35 as default for the overlapping area ratio between the bboxes of detected object and ground truth"\
    && exit 1
fi
bgobjects=$1
denormalize=$2
reduced=$3
min_overlap=${4:-0.35}

WORKDIR=$(dirname `readlink -f $0`)

CORP=/home/data-store/corpora/bgobjects
base_indices=${CORP}/indices
base_results=${CORP}/SRC/results

if [[ $bgobjects -eq "1" ]]; then
    base_labels=${CORP}/SRC/bgobjects/vabels
	python buddy_evaldet.py "$base_results/" \
                                "$base_indices/val.txt" \
                                "$base_labels/" $denormalize $reduced $min_overlap
else
    base_ref=${CORP}/SRC/ref_data
	python ${CORP}/tools/Code/utils/buddy_evaldet.py "$base_results/" \
                               "$base_ref/val.txt" \
                                "$base_ref/labels/" $denormalize $reduced $min_overlap
fi
