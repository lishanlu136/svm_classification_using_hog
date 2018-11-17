#include "test.h"

void ComputeHogFeatures(Mat &roi, vector<float> &features)
{
	//灰度化
	Mat grayImage;
	cvtColor(roi,grayImage,CV_BGR2GRAY);
	//gamma矫正并归一化
	Mat gammaImage;
	grayImage.convertTo(gammaImage,CV_32F);  //转换为浮点型
	sqrt(gammaImage,gammaImage);
	//normalize(gammaImage,gammaImage,0,255,NORM_MINMAX,CV_8UC1);
	//计算方向梯度
	Mat gradient  = Mat::zeros(gammaImage.rows,gammaImage.cols,CV_32F);
	Mat theta = Mat::zeros(gammaImage.rows,gammaImage.cols,CV_32F);
	for (int i=1;i<gammaImage.rows-1;i++)
	{
		for (int j=1;j<gammaImage.cols-1;j++)
		{
			float gx,gy;
			gx = gammaImage.at<float>(i,j+1)-gammaImage.at<float>(i,j-1);
			gy = gammaImage.at<float>(i+1,j)-gammaImage.at<float>(i-1,j);
			gradient.at<float>(i,j) = sqrt(gx*gx+gy*gy);               //计算梯度的模
			theta.at<float>(i,j) = (float)(atan2f(gy,gx)*180/CV_PI);   //计算梯度的方向
			//cout<<"梯度："<<gradient.at<float>(i,j)<<"方向："<<theta.at<float>(i,j)<<endl;      //打印梯度和方向信息
		}
	}
	//normalize(gradient,gradient,0,255,NORM_MINMAX,CV_8UC1);
	//imshow("梯度图",gradient);
	//调整图像大小，使图像能被整数块
	resize(gradient,gradient,cvSize((int)(gradient.cols/CELL_SIZE)*CELL_SIZE,(int)(gradient.rows/CELL_SIZE)*CELL_SIZE));
	resize(theta,theta,cvSize((int)(theta.cols/CELL_SIZE)*CELL_SIZE,(int)(theta.rows/CELL_SIZE)*CELL_SIZE));
	int cellNum = (theta.cols/CELL_SIZE)*(theta.rows/CELL_SIZE);
	float **histBins;
	histBins = new float*[cellNum];
	for(int i=0;i<cellNum;i++)
	{
		histBins[i]=new float[BIN_NUM];
		for (int j=0;j<BIN_NUM;j++)
		{
			histBins[i][j] = 0;    //必须初始化，每个编译器的默认值不一样
		}
	}

	int num =0;
	for (int i=0;i<theta.rows;i=i+CELL_SIZE)
	{
		for (int j=0;j<theta.cols;j=j+CELL_SIZE)
		{
			for (int m=0;m<CELL_SIZE;m++)
			{
				for (int n=0;n<CELL_SIZE;n++)
				{
					float angel = theta.at<float>(i+m,j+n);    //范围-180到180
					float grad = gradient.at<float>(i+m,j+n);
					if ((angel>=0)&&(angel!=180))
					{
						int a = (int)(angel/ANGLE_SCALE);
						histBins[num][a] = histBins[num][a]+grad;
					}
					else if (angel<0)
					{
						angel = angel+180;
						int a = (int)(angel/ANGLE_SCALE);
						//cout<<"("<<histBins[num][a]<<","<<grad<<")"<<"\t";
						histBins[num][a] = histBins[num][a]+grad;
						//cout<<histBins[num][a]<<"\t";
					}
					else
					{
						histBins[num][BIN_NUM-1] = histBins[num][BIN_NUM-1]+grad;
					}
				}
			}
			num = num + 1;
		}
	}

	//打印histBins信息
	/*for (int i=0; i<cellNum; i++)
	{
	for (int j=0; j<BIN_NUM; j++)
	{
	cout<<"["<<histBins[i][j]<<"]"<<"\t";
	}
	}*/

	vector<float> blockFeatures;    //定义blockFeatures行向量，维度为[1][BIN_NUM*cellNum]
	for (int i=0; i<cellNum-theta.cols/CELL_SIZE+1; i+=(theta.cols/CELL_SIZE)*(BLOCK_SIZE/CELL_SIZE))
	{
		for (int j=0; j<theta.cols/CELL_SIZE; j+=(BLOCK_SIZE/CELL_SIZE))
		{
			for (int m=0; m<(BLOCK_SIZE/CELL_SIZE); m++)
			{
				for (int n=0; n<(BLOCK_SIZE/CELL_SIZE); n++)
				{
					for (int k=0; k<BIN_NUM; k++)
					{
						blockFeatures.push_back(histBins[i+m*(theta.cols/CELL_SIZE)+j+n][k]);
					}
				}
			}
		}
	}

	//copy(blockFeatures.begin(),blockFeatures.end(),ostream_iterator<float>(cout,"\t"));   //打印vector向量
	//归一化blockFeatures
	float temp[BIN_NUM*(BLOCK_SIZE/CELL_SIZE)*(BLOCK_SIZE/CELL_SIZE)];
	
	for (int i=0; i<blockFeatures.size(); i=i+(BIN_NUM*(BLOCK_SIZE/CELL_SIZE)*(BLOCK_SIZE/CELL_SIZE)))
	{
		float sum =0;
		for (int j=0; j<BIN_NUM*(BLOCK_SIZE/CELL_SIZE)*(BLOCK_SIZE/CELL_SIZE); j++)
		{
			temp[j] = blockFeatures[i+j];
			sum = sum + temp[j]*temp[j];
		}
		for (int k=0; k<BIN_NUM*(BLOCK_SIZE/CELL_SIZE)*(BLOCK_SIZE/CELL_SIZE); k++)
		{
			features.push_back(temp[k]/sqrt(sum));
		}
		
	}
	//释放二维动态数组内存
	for(int i=0;i<cellNum;i++)
	{
		delete []histBins[i];
	}
	delete []histBins;
}

bool get_filelist_from_dir(string path, vector<string>& files)
{
	long   hFile   =   0;
	struct _finddata_t fileinfo;
	files.clear();
	if((hFile = _findfirst(path.c_str(), &fileinfo)) !=  -1)
	{
		do
		{
			if(!(fileinfo.attrib &  _A_SUBDIR))
				files.push_back(fileinfo.name);
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
		return true;
	}
	else
		return false;
}