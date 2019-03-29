// Example.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <io.h> 
#include <direct.h> 
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#define GroundTruthNum 1000

struct Boundingbox
{
	int nCam;
	int nFrm;
	int ObjID;
	CvRect r;
};

void WriteOutput(Boundingbox* output, int length, FILE *fp)
{
	int i;
	for(i=0;i<length;i++)
	{
		fprintf(fp,"%d,%d,%d,%d,%d,%d,%d\n",output[i].nCam,output[i].nFrm,output[i].ObjID,output[i].r.x,output[i].r.y,output[i].r.width,output[i].r.height);
	}
}

int Example_Tracker(Boundingbox* output, FILE* inputfile, FILE* datasetfile)
{
	FILE *resultfile = fopen("result.dat", "r");
	int i;
	for(i=0;feof(resultfile)==0;i++)
	{
		fscanf(resultfile,"%d %d %d %d %d %d %d",&output[i].nCam,&output[i].nFrm,&output[i].ObjID,&output[i].r.x,&output[i].r.y,&output[i].r.width,&output[i].r.height);
	}
	int length = i-1;
	fclose(resultfile);
	return length;
}

int _tmain(int argc, _TCHAR* argv[])
{
	Boundingbox output[GroundTruthNum];

	FILE *inputfile = fopen("input_groundtruth.txt", "r");
	FILE *datasetfile = fopen("dataset.txt", "r");

	if (!inputfile) {
		fprintf(stderr, "Input groundtruth file (input_groundtruth.txt) not available. Stopping.\n");
		exit(-1);
	}

	if (!datasetfile) {
		fprintf(stderr, "Video list file (dataset.txt) not available. Stopping.\n");
		exit(-1);
	}

	//please replace by your algorithm
	int length = Example_Tracker(output,inputfile,datasetfile);
	//algorithm done

	FILE *outputfile = fopen("output.txt", "w");
	WriteOutput(output, length, outputfile);
	fclose(outputfile);


	return 0;
}

