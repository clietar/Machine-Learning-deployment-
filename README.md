# Machine-Learning-deployment
Tools and methods to deploy ML models in production 

<b> XGBOOST </b>

- <strong>script "XGB_json_reader.py" </strong> was realised to deploy a trained Xgboost model in pure python. 

    1. objective : remove the dependencies of xgboost library function when using the xgboost.predict() method over new observations converted into XGBOOST format (using .DMatrix()). To sum it up, it ables to deploy a machine learning solution customized with XGBOOST, without XGBOOST when predicting.
    
    2. principle : 
        - instead of dumping trained model as a pickle and load it for deployment, we dump the model as .json file using the dump_model() function of XGBOOST. The file contains a list of python dictionaries, each representing the boosters created during the training phase. 
        - the script first represents each node (its respective children and leaves) and each decision treee as class objects from the .json file
        - Finally, for the given input data, it will logically read every booster representations to get the right output value, and make prediction as XGBOOST algorithm does.
      
     3. Utilization : 
        - Store the trained model using the BoosterReader class and specify if its a classification or regression problem : model = BoosterReader(your .json file, "classification" or "regression", your base_score initialization)
        - Call the .predict() function on new observations. 
       
     4. Requirements :
        - before dumping model for a classification problem, base_score must be the default value (0.5)
        - the model must be dumped to a .json file using a feature map (.txt file) of your variables. Feature names must not contain spaces
        - for prediction, input data must be passed as a list of dicts (if using a pandas dataFrame, use .to_dict(orient='records') method. 
