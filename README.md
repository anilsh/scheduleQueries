
# ScheduleQueries
** Matlab implementation of ICAPS 2019 paper titled 'Reinforcement Learning based Querying in Camera Networks for Efficient Target Tracking'

_Anil Sharma, Saket Anand, Sanjit K. Kaul_

[[Paper](https://www.iiitd.edu.in/~anils/)] [[Video](https://www.youtube.com/watch?v=Yn0AVk7wEyI&feature=youtu.be)] [[BibTeX](https://github.com/anilsh/scheduleQueries/blob/master/sharma_icaps2019.bib)]

---
Target Tracking in a Camera Networks is the problem of determining the location of the target at all times as the target moves in different camera field of views. The input is the video sequence  of all cameras and the initial location of the target and the output should be the trajectory of the target whose each instance contain (c,x,y), where c is the camera index and (x,y) is the spatial location (or bounding box) in the camera c. 

In general, the related methods solve this problem in a two-step framework. First, SCT (single camera tracking) is applied to identify the trajectory of the target within a single camera FOV. Second, ICT (inter-camera tracking) is performed to resolve handovers by finding associations for the target tracked by SCT. This association is computationally expensive. We proposed an intelligent method for scheduling cameras of the person trajectory using reinforcement learning. We have shown that our method subtantially reduces the number of frames required to be processed at the cost of a small reduction in recall. We have used NLPR MCT dataset, which is a real multi-camera multi-target tracking benchmark. 

In brief, our method is also two-step (but can be run online). First, it selects a camera index where the target is likely to be present. Second, in the selected camera it finds the target using re-identification. We didn't use re-identification but simulated errors in a typical re-identification pipeline.

In this repository, we provide MATLAB code to run and evaluate our method. This code has been written over the past years as part of my PhD research. The MCT literature proposed all vision based method but we didn't cater to the vision challenges (particularly in this paper).

---

## Code

The code is organized as follows:
1. ```main_q_dbx_icaps.m``` is the main file to call training and testing function for a sub-dataset x. Please note that sub-dataset-1 and 2 share same network topology and hence only one model is trained for both. 
2. All folder names are self-explanatory and you don't need to edit any of the file inside any folder to reproduce the results.

## Downloading the data

### NLPR MCT benchmark dataset

Download the dataset from the [[official website](http://mct.idealtest.org/Datasets.html)] . This dataset contains four sub-datasets and we have used all four in our method.

---

## Running the tracker

Edit the variable ```task``` in the main file of a particular sub-dataset (```task=2``` is for testing with ground truth re-id). To simulate errors in re-identification, change ```task``` and edit ```opts.fpath``` in the main file to point the location of the ground-truth file.


### Dependencies

The code should run with basic matlab functions and doesn't require any sophisticated installation. 

### Pre-trained model

The pre-trained model and results for each specific case of the simulation are provide in 'model' and 'results_icaps' directories respectively. 

### Tables and Figures
In folder 'metric', you will find various scripts to reproduce all tables and figures reported in the paper. Use [MCT evaluation kit](http://mct.idealtest.org/Datasets.html) to generate MCTA values from the results file. Result files are kept in ```results_icaps``` folder for every case. 


### Training

To train a model from scratch, set ```task=0``` in the main file. Edit epochs, the provided model was trained for 50K epochs. Since it is a table based method and it is not possible to include all possible cases so we have set maximum value of the ```telapse``` to be 200 and the policy behaves randomly after reaching this value.

### Visualization

To reproduce figure 4 of the paper. Edit the scripts in the train folder and uncomment following lines. Set first argument of subplot accordingly. 
```
    subplot(length(pALL(2:2:end)),1,rs_cnt);
    a= [(repmat(gt_cam_allt',100,1)); zeros(5,length(gt_cam_allt)); (repmat(pr_cam_allt',100,1))];
    colormap('hot');
    imagesc(a);
    set(gca,'ytick',50:100:200)
    set(gca,'yticklabel',{'GT','Sel'})

```


## Remarks

The related method are multi-camera multi-target and our method is multi-camera single target tracking. For fair comparison, we have run multiple parallel iterations of our method one for each target of the test set to make it multi-camera multi-target approach.


## Extension and improvement in results

We have extended the proposed approach by including random jumps in the exploration of the target's trajectory during training. This has provided significant improvement in accuracy, precision and recall. The code is available on request. 

We have also used a re-identification implementation  of [[this paper](https://github.com/layumi/Person_reID_baseline_matconvnet)] to replace ```isTargetPresent()``` function used in training and testing functions. We are not providing the code but you are encouraged to use the linked implementation. 


If this code helps your research, please cite the following work which made it possible.

```
@inproceedings{sharmaScheduleCameras,
  title =        {Reinforcement Learning based Querying in Camera Networks for Efficient Target Tracking},
  author =       {Sharma, Anil and Anand, Saket and Kaul, Sanjit},
  booktitle =    {ICAPS},
  year =         {2019}
}

```

## License

This code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Some external dependencies have their own license.

