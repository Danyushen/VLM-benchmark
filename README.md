# VLM-benchmark
This repository provides a script to evaluate MiniCPM-Llama3-V 2.5 int4 (https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4) on GQA benchmark, which contains 113k images from COCO and Flickr (paper: https://arxiv.org/pdf/1902.09506v3). Note that the model and the listed requirements are tested on python 3.10. 

## Download data 
The data can be downloaded from this link: https://cs.stanford.edu/people/dorarad/gqa/download.html, by simply clicking on the three buttons on the top 'Download Scene Graphs', 'Download Questions', and 'Download Images' (20.3 GB). When they are downloaded, extract the files and move them to the data folder. When this is done, there should be the following folders in data: images, questions, and sceneGraphs. In total, it requires 39.8 GB of space. Note that the data from folder sceneGraphs is not used. 

## Test the model 
The file gqa_minicpm.py evaluates MiniCPM-Llama3-V 2.5 int4 on a subset of GQA benchmark. The subset consist of all images from testdev_balanced_questions.json that has a yes/no question. (testdev is a small subset of GQA for development, the 'balanced' version contains a subset of the questions in the 'all' version and is less biased. Details can be found in /data/questions/readme.txt)

Two json files will be stored in /test_results:
- All predictions together with their corresponding question_id
- The wrong predictions together with their corresponding image_id, question, and correct answer

The accuracy is computed by comparing yes/no in predictions and answers. 
