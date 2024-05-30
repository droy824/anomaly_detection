# Dependencies
This code runs on:
* Python 3
* OpenCV (cv2 version 4.8.0)
* NumPy
* sklearn
* TensorFlow (>=1.0)
* PyTorch
* MatPlotLib

# Running the code
## Directories and files
The following directories and files are present in `anomaly_detection`: 
1. `DeepSORT`: Cloned repo from https://github.com/nwojke/deep_sort/tree/master used to create detection dataset.

2. `det`: This directory contains detection files and class files (classes of deteced objects) from YOLOv3. These detection files are fed to DeepSORT algorithm.

3. `Inputs`: This directory contains detection files from the DeepSORT algorithm. These files are used as input to train the LSTM.

4.  `nanonets_object_tracking-master`: Cloned repo from https://github.com/abhyantrika/nanonets_object_tracking/ used to create detection dataset.

5. `Test`: This directory contains detection files from the DeepSORT algorithm. These files are used as input to test the LSTM.

6. `YOLOv3`: Cloned repo from https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master used to create detection dataset.

7. `Making Dataset v2.ipynb`: Used to create the training dataset for the LSTM. It takes videos batch by batch from the attached zip files, and runs object detection (using YOLOv3) and object tracking (using NanoNets DeepSORT) on them to generate detections. The resulting dataset of detections consists of `frame_num, track_id, x_min, y_min, x_max, y_max, class, anomaly_in_frame`. Here, `frame_num` refers to the frame, `track_id` refers to the object ID, `x_min, y_min, x_max, y_max` refer to the bounding box coordinates, `class` refers to the YOLOv3 object class (0 being person, etc.), `anomaly_in_frame` being the anomaly class in frame (0 being 'Abuse', 1 being 'Assault', etc.).

8. `Making test dataset.ipynb`: Used to create the test dataset to evaluate the LSTM. Similar to `Making Dataset v2.ipynb`, it takes videos from the `test_dataset_2.zip` file and creates a detection dataset.

9. `Anomaly Detection.ipynb`: Used to train the LSTM. It consists of custom datasets, custom DataLoaders, LSTM training loop and testing loop.

10. `model_1.pth`: Pre-trained weights of the LSTM after training for 10 epochs.

11. `model_2.pth`: Pre-trained weights of the LSTM after training for 20 epochs.

12. `model_3.pth`: Pre-trained weights of the LSTM after training for 30 epochs.

13. `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`: File with temporal annotations for training and testing videos - tells us which frame contains which anomaly and in which video.

14. `Train`: Contains training videos.

15. `Test_dataset_2.zip`: Contains testing videos.

## Executing code
1. First, create the training dataset by running all code cells in `Making Dataset v2.ipynb`. Ensure the program is run for batches 1-9. This can be done by changing the batch number in the code cells in the notebook.

2. Next, create the testing dataset by running all code cells in `Making test dataset.ipynb`.

3. Finally, create the custom datasets and DataLoaders, and train the LSTM with `Anomaly Detection.ipynb`. 

To check training accuracy, change line 13 in `extract_test_anomaly_type()` definition from:
```
filename = filename[5:]
```
to 
```
filename = filename[6:]
```
Next, re-run the definition of `AnomalyTestDataset` and evaluate training accuracy.

# Credit
This code follows implementations of YOLOv3 (https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master) and NanoNets DeepSORT (https://github.com/abhyantrika/nanonets_object_tracking/), and uses the UCF-Crime dataset (https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf).