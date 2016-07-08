#include <caffe/caffe.hpp>
#include <caffe/util/yolo_utils.hpp>
#include <caffe/data_transformer.hpp>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#ifdef USE_OPENCV
using std::string;

string get_file_name(const string& s) {

   char sep = '/';

   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      string file_name = s.substr(i+1, s.length() - i);
      size_t lastindex = file_name.find_last_of(".");
      return file_name.substr(0, lastindex);
   }
   return("");
}

void print_yolo_detections(std::vector<string>& eval_result_files, string file_id,
		caffe::YoloBox *boxes, float **probs, int total, int classes, int w, int h)
{
    for(int i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].width/2.;
        float xmax = boxes[i].x + boxes[i].width/2.;
        float ymin = boxes[i].y - boxes[i].height/2.;
        float ymax = boxes[i].y + boxes[i].height/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(int j = 0; j < classes; ++j){
            if (probs[i][j]){
            	std::ofstream outfile(eval_result_files[j].c_str(), std::ios::app);
            	if(outfile)
            		outfile <<file_id<<" "<<probs[i][j]<<" "<<xmin<<" "<<ymin<<" "<<xmax
            			<<" "<<ymax<<std::endl;
            }
        }
    }
}

class YoloPredictor{
public:
	YoloPredictor(const string& eval_config_file);

	void Validate();

private:
	typedef struct {
		float prob_threshold;
		float iou_threshold;
		int grid_width;
		int boxes_per_cell;
		int classes;
		bool square;
		bool only_objectness;
		std::vector<std::string> labels;
		std::string eval_data_prefix;
	 } YoloConfig;

	caffe::Net<float> *net_;
	caffe::TransformationParameter transform_param_;
	caffe::DataTransformer<float> *data_transformer_;
	YoloConfig yolo_config_;
	caffe::YoloBox *boxes_;
	float **probs_;
	std::vector<std::string> eval_files_;
	std::vector<std::string> eval_result_files_;
	int num_boxes_;
};

YoloPredictor::YoloPredictor(const string& eval_config_file){

	cv::FileStorage fs(eval_config_file, cv::FileStorage::READ);
	if (!fs.isOpened())
		throw std::runtime_error("Could not open eval config file");

	cv::FileNode fn = fs.root();

    string model_file;
	string trained_file;
	string eval_data_file;

	fn["model_file"] >> model_file;
	fn["trained_file"] >> trained_file;
	fn["eval_data_file"] >> eval_data_file;
	fn["prob_threshold"] >> yolo_config_.prob_threshold;
	fn["iou_threshold"] >> yolo_config_.iou_threshold;
	fn["classes"] >> yolo_config_.classes;
	fn["grid_width"] >> yolo_config_.grid_width;
	fn["boxes_per_cell"] >> yolo_config_.boxes_per_cell;
	fn["square"] >> yolo_config_.square;
	fn["only_objectness"] >> yolo_config_.only_objectness;
	fn["eval_data_prefix"] >> yolo_config_.eval_data_prefix;

	cv::FileNodeIterator itr = fn["labels"].begin();
	cv::FileNodeIterator itrEnd = fn["labels"].end();
	for (; itr != itrEnd; ++itr)
		yolo_config_.labels.push_back(static_cast<std::string>(*itr));

  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  net_ = new caffe::Net<float>(model_file, caffe::TEST);
  net_->CopyTrainedLayersFrom(trained_file);

  transform_param_.set_scale(1/255.0);

  data_transformer_ = new caffe::DataTransformer<float>(transform_param_, caffe::TEST);

  num_boxes_ = yolo_config_.grid_width * yolo_config_.grid_width *
		  yolo_config_.boxes_per_cell;
  boxes_ = new caffe::YoloBox[num_boxes_];
  probs_ = new float *[num_boxes_];

  for (int i = 0; i < num_boxes_; ++i) {
	  probs_[i] = new float[yolo_config_.classes];
  }

  // List of evaluation images
  std::ifstream infile(eval_data_file.c_str());
  CHECK(infile.good()) << "Failed to open evaluation data file "
  				<< eval_data_file;
  for(std::string line; std::getline(infile, line);){
	  eval_files_.push_back(line);
  }

  // Class specific eval data files
  for(int i = 0; i < yolo_config_.labels.size(); ++i){
	  std::string file_name = yolo_config_.eval_data_prefix +yolo_config_.labels[i]+".txt";
	  eval_result_files_.push_back(file_name);
	  std::ofstream outfile;
	  outfile.open(file_name.c_str());
  }

}

void YoloPredictor::Validate(){

	cv::Mat cv_img, resized_img;
	int batch_size = net_->input_blobs()[0]->num();
	std::cout << "Starting validation with batch size " << batch_size << std::endl;
	for(int i=0; i < eval_files_.size();){

		std::vector<int> input_shape(4);
		input_shape[0] = 1;
		input_shape[1] = net_->input_blobs()[0]->channels();
		input_shape[2] = net_->input_blobs()[0]->height();
		input_shape[3] = net_->input_blobs()[0]->width();
		if((i + batch_size) > eval_files_.size()){
			net_->input_blobs()[0]->Reshape(input_shape);
			net_->Reshape();
			batch_size = 1;
			std::cout << "Switching to batch size " << batch_size << std::endl;
		}
		float* input_data = net_->input_blobs()[0]->mutable_cpu_data();
		caffe::Blob<float> transformed_data;
		transformed_data.Reshape(input_shape);

		std::vector<int> org_width(batch_size);
		std::vector<int> org_height(batch_size);
		for(int j = 0; j < batch_size; ++j){

			cv_img = cv::imread(eval_files_[i+j]);
			org_width[j] = cv_img.cols;
			org_height[j] = cv_img.rows;

			cv::resize(cv_img, resized_img,
					cv::Size(net_->input_blobs()[0]->width(), net_->input_blobs()[0]->height()));

			int input_offset = net_->input_blobs()[0]->offset(j);
			transformed_data.set_cpu_data(input_data + input_offset);

			data_transformer_->Transform(resized_img, &transformed_data);
		}

		net_->Forward();

		const float* predictions = net_->output_blobs()[0]->cpu_data();
		for(int j = 0; j < batch_size; ++j){

			int offset = net_->output_blobs()[0]->offset(j);

			convert_yolo_detections(predictions+offset, probs_, boxes_,
					yolo_config_.grid_width, yolo_config_.boxes_per_cell,
					yolo_config_.classes, yolo_config_.square,
					org_width[j], org_height[j], yolo_config_.only_objectness,
					yolo_config_.prob_threshold);

			non_maximum_suppression(boxes_, probs_, num_boxes_, yolo_config_.classes,
					yolo_config_.iou_threshold);

			string file_id = get_file_name(eval_files_[i+j]);

			print_yolo_detections(eval_result_files_, file_id, boxes_, probs_,
					num_boxes_, yolo_config_.classes, org_width[j], org_height[j]);
		}

		i += batch_size;
		std::cout << "Validated " << i << " of " << eval_files_.size() << " images" << std::endl;
	}
}

int main(int argc, char** argv) {

	if (argc != 2) {
    std::cerr << "Usage: " << argv[0]
              << " yolo_eval_config.xml" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  const string eval_config_file = argv[1];

  YoloPredictor yolo_predictor(eval_config_file);

  yolo_predictor.Validate();

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "Requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


