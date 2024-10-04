# Thai_and_English_OCR Yiyi Wang
## Intro: Directory structure of datasets  
### Part of structure 

Before starting to split datasets, I think it's important to understand the structure of directory.
<ul><li>ThaiOCR
  <ul><li>ThaiOCR-TrainigSet
    <ul><li>English<ul><li>046(number)<ul>
      <li>200(dpi)<ul>
        <li>bold(style)<ul>bmp files</ul></li>
        <li>bold_italic<ul>bmp files</ul></li>
        <li>italic<ul>bmp files</ul></li>
        <li>normal<ul>bmp files</ul></li>
      </ul></li>
      <li>300</li>
      <li>400</li>
    </ul>
    </li></ul>
    </ul></li>
  </li>
    <li>ThaiOCR-TestSet</li></ul></li></ul>
</li></ul>

## Generate training, test, and validation samples (splitBMP.py)
#### Arguments explanation
 ```
 '--input_folder' -> 'Folder path containing the BMP file'
 '--output_folder' -> 'Output folder path to save the generated samples'
 '--language' -> 'Select recognition language including Thai and English for training set'
 '--dpi' -> 'DPI of the images to filter for training set'
 '--style' -> 'Font style of the images to filter for training set'
 '--test_language' -> 'Select recognition language including Thai and English for test set'
 '--test_dpi' -> 'DPI of the images to filter for test set'
 '--test_style' -> 'Font style of the images to filter for test set'
 ```
#### Example of Usage
 ```python
 python splitBMP.py --input_folder /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/ --output_folder /home/guswanyie@GU.GU.SE/srv/www/ --language thai --dpi 400 --style normal --test_language english --test_dpi 300 --test_style bold
 ```
#### Results
This script generated three datasets according to arguments setting input by user. These sets are train, test, and val.

## Train a model on a training set generated from previous script (trainBMP.py)
Firstly, there are no labels for bmp files, so I extract file name to generate unique labels for training sets. Also, for dataset preprocessing, I convert image sets to grayscale. For neural network, I use CNN model to output the trained model 'thaiocr_model.pth' and I use cuda:0 on mltgpu server.
 ```python
 # Some settings of training model.
 batch_size = 32
 num_epochs = 10
 learning_rate = 0.001
 # Run the script for training
 python trainBMP.py
 ```
#### Results
Run the script and get the model 'thaiocr_model.pth'.
## Statistics analysis (testBMP.py)
|       | training data      | testing data | precision     |recall | F1 |accuracy|
| :---        |    :----:   |   :----:   |    :----:   |  :----:   | :----:   |         ---: |
| 1   | Thai normal text, 200dpi   | Thai normal text, 200dpi    | 0.9604   |0.9640   |0.9634   | 0.9640      |
| 2   | Thai normal text, 400dpi   | Thai normal text, 200dpi    | 0.9995     | 0.9962     | 0.9993     |0.9962    |
| 3   | Thai normal text, 400dpi  | Thai bold text, 400dpi   | 0.9995     | 0.9962     | 0.9993     |0.9962    |
| 4   | Thai bold text  | Thai normal text   | 0.9317     | 0.9421     | 0.9302     |0.9421    |
| 5   | All Thai styles  | All Thai styles   | 0.9982     | 0.9994     | 0.9995     |0.9994    |
| 6   | Thai and English normal text jointly  | Thai and English normal text jointly   | 0.9991     | 0.9974     | 0.9993     |0.9974    |
| 7   | All Thai and English styles jointly  | All Thai and English styles jointly   | 0.9990     | 0.9981     | 0.9992     |0.9981    |
## Challenges and Observation
In this assignments, I met with lots of error and bugs. I would like to share two of them. First one is AttributeError: 'tuple' object has no attribute 'to'. The reason of this error is because my labels should be integers before passing them to the model, as nn.CrossEntropyLoss() expects labels to be class indexed. If my label is a string type, I need to convert it to an integer. So I fix it by using LabelEncoder.
The second is that 'ValueError: y contains previously unseen labels: 'Asts332'', which means the label in validation is inconsistent, and unknown labels appear. Also, since models will never predict a label that wasn't seen in their training data, LabelEncoder should never support an unknown label. Here is my answer but did not work. Finally, I selected 'label_encoder.fit_transform()' method as an instead solution.
```python
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_encoder.fit(train_labels.reshape(-1, 1))
ordinal_encoder.transform(val_labels.reshape(-1, 1))
ordinal_encoder.transform(val_labels.reshape(-1, 1))
```

Moreover, when doing this assignment, I realize that when we coding in command line instead of a user friendly GUI, which means that it's not easy to debug our code, we need to manually add some breakpoint. For example, for the split script, we would better to add 'print(dir_path)' in some loops to see whether everything goes correctlly. Also for training, we can add 'print(f'{epoch}', end='\r')' to monitor which round the training is going into.
