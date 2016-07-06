'''
Created on Apr 27, 2016

@author: karthik.govindappa
'''

import os
import xml.etree.ElementTree as ElementTree
import numpy as np
import argparse
import pickle

class InitParams(object):
    '''
    Initializes parameters and necessary file paths of VOC dataset 
    '''

    def __init__(self, config_file):
        '''
        Constructor
        '''
        element_tree = ElementTree.parse(open(config_file))
        config_file_root = element_tree.getroot()
        
        # dataset
        self.__dataset = config_file_root.find('dataset').text 
        
        # change this path to point to your copy of data
        self.__datadir = config_file_root.find('datadir').text 
                
        # change this path to a writable local directory for the example code
        self.__localcache = self.__set_localcache(self.__datadir, self.__dataset)
                
        # initialize main challenge paths
        self.__annopath = config_file_root.find('annopath').text
        self.__imgsetpath = config_file_root.find('imgsetpath').text
        self.__detrespath = config_file_root.find('detrespath').text
        
        # initialize the VOC challenge options
        # classes
        self.__classes = config_file_root.find('classes').text.split()

        # overlap threshold
        self.__minoverlap = float(config_file_root.find('minoverlap').text)
        
        # annotation cache for evaluation
        self.__annocachepath = self.__set_annocachepath(self.__localcache)
        
               
    def __get_dataset(self):
        return self.__dataset
    
    def __get_datadir(self):
        return self.__datadir
    
    def __set_localcache(self, root, dataset):
        return os.path.join(root, "cache")
    
    def __get_localcache(self):
        return self.__localcache
    
    def __get_annopath(self):
        return self.__annopath
        
    def __get_imgsetpath(self):
        return self.__imgsetpath
    
    def __get_detrespath(self):
        return self.__detrespath
    
    def __get_classes(self):
        return self.__classes
    
    def __get_minoverlap(self):
        return self.__minoverlap
    
    def __set_annocachepath(self, localcache):
        return os.path.join(localcache, "{}_anno.pkl".format(self.__dataset))
    
    def __get_annocachepath(self):
        return self.__annocachepath
        
    dataset = property(__get_dataset, None, None, None)
    datadir = property(__get_datadir, None, None, None)       
    localcache = property(__get_localcache, None, None, None)
    annopath = property(__get_annopath, None, None, None)
    imgsetpath = property(__get_imgsetpath, None, None, None)
    detrespath = property(__get_detrespath, None, None, None)
    classes = property(__get_classes, None, None, None)
    minoverlap = property(__get_minoverlap, None, None, None)
    annocachepath = property(__get_annocachepath, None, None, None)
    
class ObjectParams(object):
    """
    Class for storing parameters of all objects in a single image 
    """        
    def __init__(self):
        self.class_names = []
        self.difficult = []
        self.bboxes = []
         
class GTObjects(object):
    """
    Class for storing parameters of given class objects, and the 
    detection status in a single image 
    """
    def __init__(self):
        self.bboxes = []
        self.difficult = []
        self.det = []

class InvalidDatasetError(Exception):
    """
    Class for raising invalid dataset exception
    """
    def __init__(self, dataset_name):
        error_msg = "The dataset '{}' is invalid".format(dataset_name)
        super(InvalidDatasetError, self).__init__(error_msg)
    
class EvaluateDetection(object):
    """
    Evaluates the trained trained detection model. Average Precision is the quantitative
    measure used here.
    
    Attributes:
        __init_params: Object containing initialization parameters of given dataset
        __model: Model to be evaluated
        __cls: Class on which model evaluation is done
        __number_of_class_objs: Number of specified class objects present in given dataset 
        __gtids: Ground truth ids
        __objects: All annotated objects
        __gt_objects: All annotated objects of given class
        __det_ids: Image ids of detections returned by the model
        __confidences: Confidence scores
        __true_positives: Precise detections made by the model 
        __false_positives: False objects detected  as positive by the model   
        __precision: Precision value
        __recall: Recall value
        ap: Average precision value
    """

    def __init__(self, config_file, cls):
        '''
        Constructor
        '''
        self.__init_params = InitParams(config_file)
               
        self.__cls = cls
        
        print "\nComputing AP for class '{}'".format(self.__cls)       
        self.__number_of_class_objs = 0
        
        # Load ground truth ids
        self.__gtids = self.__load_ground_truth_ids()
        print '\tLoading ground truth ids...done!'
        
        # Load annotated objects from xml files
        self.__objects = self.__load_annotated_objects(self.__gtids)
        print '\tLoading annotations...done!'
        
        # Isolate all objects corresponding to given class
        self.__gt_objects = self.__filter_class_specific_objects(self.__objects, 
                                                                 self.__cls)
        print '\tFilter class specific annotations...done!'
        
        # Load detection results provided by the trained model 
        self.__det_ids, self.__confidences, self.__bboxes = \
        self.__load_detection_results(self.__cls)
        print '\tLoading detection results...done!'
        
        # Compute true and false positives
        self.__true_positives, self.__false_positives = \
        self.__assign_detections_to_gtobjects(self.__det_ids, self.__confidences,
                                              self.__bboxes, self.__gt_objects )
        print '\tValidating detections...done!'
        
        # Compute precision/recall
        self.__recall, self.__precision = self.__compute_precision_recall(self.__true_positives, 
                                                                          self.__false_positives)
        print '\tComputing precision/recall...done!'
        
        # Compute average precision
        self.ap = self.__compute_avg_precision(self.__recall, self.__precision)
        print '\tComputing average precision...done!'
        
    def __load_ground_truth_ids(self):
        """ Load ground truth ids from the initialized file path """
        gt_ids_file_path = self.__init_params.imgsetpath
        gtids = np.genfromtxt(gt_ids_file_path, dtype=np.str)
        return gtids
    
    def __load_annotated_objects(self, gtids):
        """ Load parameters of all objects in every image's annotated xml file """
        all_parsed_objects = dict()
        if os.path.isfile(self.__init_params.annocachepath):
            with open(self.__init_params.annocachepath, 'rb') as handle:
                all_parsed_objects = pickle.load(handle)
        else:
            if not os.path.isdir(self.__init_params.localcache):
                os.mkdir(self.__init_params.localcache)
            for gtid in gtids:
                annotated_file_path = self.__init_params.annopath.format(gtid)
                parsed_objects = self.__parse_annotation_file(annotated_file_path)
                all_parsed_objects[gtid] = parsed_objects
            with open(self.__init_params.annocachepath, 'wb') as handle:
                pickle.dump(all_parsed_objects, handle)
        return all_parsed_objects
            
    def __parse_annotation_file(self, file_path):
        """ Parse xml file to retrieve object parameters"""    
        parsed_objs = ObjectParams()
        element_tree = ElementTree.parse(open(file_path))
        root = element_tree.getroot()
        objs = root.findall('object')
        for obj in objs:
            parsed_objs.class_names.append(obj.find('name').text)
            parsed_objs.difficult.append(int(obj.find('difficult').text))
            bbox_obj = obj.find('bndbox')
            bbox = []
            bbox.append(int(bbox_obj.find('xmin').text))
            bbox.append(int(bbox_obj.find('ymin').text))
            bbox.append(int(bbox_obj.find('xmax').text))
            bbox.append(int(bbox_obj.find('ymax').text))
            parsed_objs.bboxes.append(bbox)
        return parsed_objs
            
    
    def __filter_class_specific_objects(self, objs_dict, class_name):
        """ Get all objects related to given class"""
        npos = 0
        gt_objects = dict()
        for gtid in objs_dict.keys():
            gt_obj = GTObjects()
            obj = objs_dict[gtid]
            names = np.array(obj.class_names)
            class_idxs = np.where(names==class_name)[0]
            gt_obj.bboxes = np.array(obj.bboxes)[class_idxs]
            gt_obj.difficult = np.array(obj.difficult)[class_idxs]
            gt_obj.det = np.zeros(len(class_idxs))
            gt_objects[gtid] = gt_obj
            # Consider only non difficult objects            
            npos += len(gt_obj.difficult) - np.count_nonzero(gt_obj.difficult)
        self.__number_of_class_objs = npos
        return gt_objects
    
    def __load_detection_results(self, class_name):
        """ Load detection results of given class"""
        file_path = self.__init_params.detrespath.format(class_name)
        results = np.genfromtxt(file_path, dtype=np.str)

        ids = results[:, 0]
        confidence = results[:, 1]
        bboxes = results[:,2:6]
        
        # cast to float
        confidence = confidence.astype(np.float)
        bboxes = bboxes.astype(np.float)
        
        # sort indices based on confidence scores in decreasing order
        desc_order_idxs = np.argsort(confidence)[::-1]
        
        # rearrange confidence,  ids and bbboxes
        ids = ids[desc_order_idxs]
        confidence = confidence[desc_order_idxs]
        bboxes = bboxes[desc_order_idxs]
        
        return ids, confidence, bboxes
        
    def __assign_detections_to_gtobjects(self, ids, confidences, bboxes, 
                                         gt_objects):
        """ Compute overlap area and estimate true and false positives"""
        true_positives = [0] * len(ids)
        false_positives = [0] * len(ids)
        
        for pos in xrange(len(ids)):
            idx = ids[pos]
            bbox = bboxes[pos]
            gt_obj = gt_objects[idx]
            
            overlap_max = 0
            pos_overlap_max = 0
            
            # compute overlap area
            for gt_pos, gt_bbox in enumerate(gt_obj.bboxes):
                new_mins = np.maximum.reduce([bbox[:2],gt_bbox[:2]])
                new_maxs = np.minimum.reduce([bbox[2:4],gt_bbox[2:4]])
                new_bbox = np.append(new_mins, new_maxs)
                # compute width and height of bounding box
                width = new_bbox[2] - new_bbox[0] + 1
                height = new_bbox[3] - new_bbox[1] + 1
                
                if (width > 0) and (height > 0):
                    # compute overlap as area of intersection / area of union
                    area_union = ((bbox[2]-bbox[0]+1) * (bbox[3]-bbox[1]+1) + 
                                  (gt_bbox[2]-gt_bbox[0]+1) * (gt_bbox[3]-gt_bbox[1]+1) - 
                                  (width * height))
                    overlap = width*height / area_union
                    # update maximum overlap area
                    if overlap > overlap_max:
                        overlap_max = overlap
                        pos_overlap_max = gt_pos
            
            # Estimate true and false positives
            if overlap_max >= self.__init_params.minoverlap:
                if not gt_obj.difficult[pos_overlap_max]:
                    if not gt_obj.det[pos_overlap_max]:
                        true_positives[pos] = 1
                        gt_obj.det[pos_overlap_max] = True
                    else:
                        false_positives[pos] = 1
            else:
                false_positives[pos] = 1  

        return true_positives, false_positives  
    
    def __compute_precision_recall(self, true_positives, false_positives):
        """ Compute precision recall"""
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recall = 1.0 * tp_cumsum / self.__number_of_class_objs 
        precision = 1.0 * tp_cumsum / (tp_cumsum + fp_cumsum)   
        
        return recall, precision
    
    def __compute_avg_precision(self, recall, precision):
        """ Compute average precision"""
        mrec = np.append([0], recall)
        mrec = np.append(mrec, [1])
        
        mpre = np.append([0], precision)
        mpre = np.append(mpre, [0])
        
        for pos in xrange(len(mpre)-2, -1, -1):
            mpre[pos] = max(mpre[pos], mpre[pos+1])
        
        idxs = [ i+1 for i,(mrec1,mrec2) in enumerate(zip(mrec[1:], mrec[:-1])) if mrec1 != mrec2 ] 
        idxs = np.array(idxs)
        
        ap = np.sum( (mrec[idxs] - mrec[idxs-1]) * mpre[idxs])         
        
        return ap
        

def main():
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("config_file", help="Xml file containing evaluation config", type=str)
    
    args = args_parser.parse_args()
    
    evaldet_person = EvaluateDetection(args.config_file, 'person')
    evaldet_cat = EvaluateDetection(args.config_file, 'cat')
    evaldet_dog = EvaluateDetection(args.config_file, 'dog')
    
    print "\n[person, cat, dog] : [{}, {}, {}] \n".format(evaldet_person.ap, evaldet_cat.ap, evaldet_dog.ap)

if __name__ == '__main__':
    
    main()
    