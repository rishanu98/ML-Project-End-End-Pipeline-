## End to End Machine Learning Project

## Dataset source https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977

**Repository Structure**:

`src/logger.py`: scripts for logging Information

`src/exception.py`: for handline exception

`src/utils.py`: To load and save the .pkl file

`src/components/data_ingesion.py`: For loading the dataset and converting train and test dataset

`src/components/data_transformation.py`: To transform and preprocess data using ML Pipeline and ColumnTransformer

`src/components/model_trainer.py`: To train and find the best model suited for the dataset

`app.py`: File contains Flask web app code

`Docker`: Docker file to build docker image name as 
[rishanu68/student_performance](https://hub.docker.com/repository/docker/rishanu68/student-performance-indicator/tags?page=1&ordering=last_updated)