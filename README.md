# radicchio
The Radicchio Triple Scorer  
  
Enviroment:  
	Python 2.7, Scikit-Learn 0.18, Numpy 1.11.1, Scipy 0.18.1   
Execution:  
	Change your current path to the src/ folder, and type in "python *.py".  
	You may first try with "python main.py".  

There are two frameworks - supervised framework and unsupervised framework.  You can inherits the abstract class SupervisedBase/UnsupervisedBase and implements different methods about feature extraction/learning/prediction to start your work. For details, please refer to the code and the comments.  

NOTE: The data is too large, so you may download it by yourself. You should have all the data files stored in the data/ folder (try ln -s !!).  

TODO:  
(1) Handle large word vectors/large names, professions, nationalities. Solution: Particiate word vectors/Build inverted index?  
(2) What's the input and output? Ask TA
(3) Need more spaces on the virtual machine  
(4) Kenall' score implementation (related to 5)  
(5) Cross validation problem: random split? or we should make validated names invisible to training?  

