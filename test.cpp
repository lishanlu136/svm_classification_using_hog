
#include "test.h"
#include "opencv2/ml/ml.hpp"

int main()
{
	
	string file_path = "E:\\car\\";
	string search_path = file_path + "*.jpg";
	vector<string> file_list;
	if (!get_filelist_from_dir(search_path, file_list))
	{
		cout << "open file error!" << endl;
	}
	vector<vector<float>> posData;
	for (int i = 0; i < file_list.size(); i++)
	{
		string image_path = file_path + file_list[i];
		Mat srcImage = imread(image_path);
		if (!srcImage.data)
		{
			printf("Can not read this image!!!");
			return -1;
		}
		//Mat roi = srcImage(Rect(20,40,16,16));
		vector<float> hogFeatures;
		ComputeHogFeatures(srcImage,hogFeatures);
		posData.push_back(hogFeatures);
	}

	//打印第10张图片的特征
	/*for (int i=0;i<data[10].size(); i++)
	{
	cout<<"["<<data[10][i]<<"]"<<" ";
	}*/

	//copy(hogFeatures.begin(),hogFeatures.end(),ostream_iterator<float>(cout,"\n"));  //打印hogFeatures信息
	//for (vector<float>::iterator it = hogFeatures.begin(); it!=hogFeatures.end(); ++it) //也可以打印hogFeatures信息
	//{
	//	cout<<*it<<endl;
	//}
	file_path = "E:\\background\\";
	search_path = file_path + "*.jpg";
	vector<string> neg_file_list;
	if (!get_filelist_from_dir(search_path, neg_file_list))
	{
		cout << "open file error!" << endl;
	}
	vector<vector<float>> negData;
	for (int i = 0; i < neg_file_list.size(); i++)
	{
		string image_path = file_path + neg_file_list[i];
		Mat srcImage = imread(image_path);
		if (!srcImage.data)
		{
			printf("Can not read this image!!!");
			return -1;
		}
		if ((srcImage.rows<64)||(srcImage.cols<64))
		{
			continue;
		}
		Mat roi = srcImage(Rect(0,0,64,64));
		vector<float> hogFeatures;
		ComputeHogFeatures(roi,hogFeatures);
		negData.push_back(hogFeatures);
	}
	//合并正负样本的特征
	vector<vector<float>> trainData;
	for (int i=0; i<posData.size(); i++)
	{
		trainData.push_back(posData[i]);
	}
	for (int i=0; i<negData.size(); i++)
	{
		trainData.push_back(negData[i]);
	}
	Mat trainDataMat = Mat(trainData.size(),trainData[0].size(),CV_32FC1);
	memcpy(trainDataMat.data,trainData.data(),trainData.size()*sizeof(float));
	//定义label
	Mat posLabels = Mat::ones(posData.size(),1,CV_32FC1);      //不能是CV_8UC1型
	Mat negLabels = Mat::ones(negData.size(),1,CV_32FC1)*(-1);
	Mat trianLabelsMat(posData.size()+negData.size(),1,CV_32FC1);
	vconcat(posLabels,negLabels,trianLabelsMat);

	//设置支持向量机的参数（Set up SVM's parameters）
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma       = 0.05;
	params.C           = 0.1;
	params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e8, 1e-6);

	//// 训练支持向量机（Train the SVM）
	CvSVM SVM;
	SVM.train(trainDataMat,trianLabelsMat,Mat(),Mat(),params);

	//SVM预测
	Mat testImage = imread("E:\\background\\backGround005144.jpg");
	if (!testImage.data)
	{
		printf("Can not read this image!!!");
		return -1;
	}
	if ((testImage.rows<64)||(testImage.cols<64))
	{
		return -1;
	}
	Mat roi = testImage(Rect(0,0,64,64));
	vector<float> testFeatures;
	ComputeHogFeatures(roi,testFeatures);
	Mat testDataMat(testFeatures);          //需要加转置才能预测
	float preLabel = SVM.predict(testDataMat.t());
	if (preLabel == 1)
	{
		cout<<"该张图片为车"<<endl;
	}
	else if (preLabel == -1)
	{
		cout<<"该张图片为背景"<<endl;
	}
	//cin.get();
	//waitKey(0);
	system("pause");
}



