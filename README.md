# Decision-Tree
Decision Tree implementation in Python, supports GINI Index and Entropy as impurity

Use the python file - trainDT.py for both training and testing.

The file trainDT.py takes the following parameters: <br>

      __train_data__ - provide the path of the file, the file should contain the training data   <br>
      __train_label__ - provide the path of the file, the file should contain the training labels  <br>
      __test_data__ - provide the path of the file, the file should contain the testing data <br>
      __test_label__ - provide the path of the file, the file should contain the testing labels  <br>
      __nlevels__  - maximum number of levels of the decision tree <br>
      __pthrd__ - threshold of the impurity  <br>
      __impurity__ - type of the impurity, currently only supports - gini index and entropy. The user should provide only either of two parameters - "gini" or "entropy" <br>
      __pred_file__ - provide the path of the file where the predictions of the test data will be stored.  <br>

Sample command: 
~~~
python3 trainDT.py -train_data="./data/train_data.txt" -train_label="./data/train_label.txt" -test_data="./data/test_data.txt" -test_label="./data/test_label.txt" -nlevels=10 -pthrd=0.2 -impurity="entropy" -pred_file="output_1.txt"
~~~
