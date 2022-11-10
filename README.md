# Applications Classifier a.k.a. Spam Classifier

*The project is aimed to detect fake applications for IT courses (spam applications) in 'email_data.xlsx' file. Data was labelled manually. 
*'data_preprocessor.py' file extracts features from the 'email_data.xlsx' file and preprocesses it (data) to be suitable as an input for a Machine Learning model. Preprocessor extract features such as applicant's name validity, phone number validity and removes redundant datathat is not used by classifier.
*'spam_detector.py' contains the classifier.
