'''
description: runing the whole learning and prediction process
parameter: USE_TEST (if TRUE using test-set, otherwise only using splitted trainingset with 4:1 and Cross-Validation with Training-Set) 
           CLASSIFIER (Classifier using for prediction) 
               -> values: SVM, DecTree, GaussProc, AdaBoost, RandForest, NeuroNet
best result: 0,7940 using SVM
'''

from iodata.readJson import readJson
from iodata.savePredJson import savePredToFile
from iodata.saveToFile import saveMatrixToCSVFile
from features.preprocessing import featureExtraction
from features.featureRanker import f_importances
from iodata.readGTData import readGT
from learning.classification import SVMclassifierTrain, classifierPredict, DecisionTreeTrain, GaussianProcessTrain, AdaBoostTrain, RandomForestTrain, NeuroNetTrain, NaiveBayesTrain
from sklearn.model_selection import train_test_split
import sklearn.metrics as scikitm
from iodata.saveToFile import saveTweetToCSVFile
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report


USE_TEST = True
CLASSIFIER = "SVM"

if __name__ == '__main__':
    
    print ("started ...")
    
    # read Training data
    dataTrain, dataHierarchyTrain = readJson("train")
    gtTrain = readGT("trainA")  
    #print(type(dataTrain))
    #print(type(gtTrain))
    print ("Training data read") 
    featureMatrixTrain, labelMatrixTrain, tweetMatrixTrain, feature_names = featureExtraction(dataTrain, gtTrain, dataHierarchyTrain,0)
    #saveMatrixToCSVFile(featureMatrixTrain,"Feature Matrix Train.csv")
    #saveMatrixToCSVFile(tweetMatrixTrain,"Tweet Matrix Train.csv")
    #saveMatrixToCSVFile(labelMatrixTrain,"Label Matrix Train.csv")
    print ("Training features extracted")
    
    
    #read Test data
    if USE_TEST:
        dataTest, dataHierarchyTest = readJson("test")
        gtTest = readGT("testA")  
        print ("Test data read")  
        featureMatrixTest, labelMatrixTest, tweetMatrixTest, feature_names = featureExtraction(dataTest, gtTest, dataHierarchyTest,1)
        print ("Test features extracted")
    
    #print (featureMatrixTrain)
    #print labelMatrixTrain
    # split Train dataset into training and testing
    
    featureMatrixTrainSplitted, featureMatrixTestSplitted, labelMatrixTrainSplitted, labelMatrixTestSplitted = train_test_split(featureMatrixTrain, labelMatrixTrain, test_size=0.2, random_state=1)

    
    # train model
    if (CLASSIFIER=="SVM"):
        model = SVMclassifierTrain(featureMatrixTrainSplitted,labelMatrixTrainSplitted,featureMatrixTestSplitted,labelMatrixTestSplitted)
    if (CLASSIFIER=="DecTree"):
        model = DecisionTreeTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="GaussProc"):
        model = GaussianProcessTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="AdaBoost"):
        model = AdaBoostTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="RandForest"):
        model = RandomForestTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="NeuroNet"):
        model = NeuroNetTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="Naive"):
        model = NaiveBayesTrain(featureMatrixTrain,labelMatrixTrain)
    
    # classify Train Testset
    predictedTrain = classifierPredict(featureMatrixTestSplitted,model)
    print ("Accuracy (Testset Train): "+str(scikitm.accuracy_score(labelMatrixTestSplitted,predictedTrain)))

    # classify test set
    if USE_TEST:
        #y = (["support","deny","query","comment"])
        #Y = label_binarize(y, classes=[0,1,2,3])
        #n_classes = Y.shape[1]
        predictedTest = classifierPredict(featureMatrixTest,model)
        #y_score = model.decision_function(featureMatrixTest)
        #print (predictedTest)
        print ("Classification report: \n", (classification_report(labelMatrixTest, predictedTest)))
        print ("Accuracy (Testset): "+str(scikitm.accuracy_score(labelMatrixTest,predictedTest)))
        print ("Precision (Testset): "+str(scikitm.precision_score(labelMatrixTest,predictedTest, average="macro")))
        print ("Recall (Testset): "+str(scikitm.recall_score(labelMatrixTest,predictedTest, average="macro")))
        print ("F1 (Testset): "+str(scikitm.f1_score(labelMatrixTest,predictedTest, average="macro")))
        #print ("Prec (Support): "+str(scikitm.precision_score(labelMatrixTest,predictedTest, average="macro", pos_label=0)))
        #print ("Prec (Deny): "+str(scikitm.precision_score(labelMatrixTest,predictedTest, average="macro", pos_label=1)))
        #precision = dict()
        #recall = dict()
        #average_precision = dict()
        #for i in range(n_classes):
        #    precision[i], recall[i], _ = precision_recall_curve(featureMatrixTest[:, i], y_score[:, i])
        #    average_precision[i] = average_precision_score(featureMatrixTest[:, i], y_score[:, i])
        #precision["micro"], recall["micro"], _ = precision_recall_curve(featureMatrixTest.ravel(),
        #         y_score.ravel())
        #average_precision["micro"] = average_precision_score(featureMatrixTest, y_score,
        #                                             average="micro")
        print (str(scikitm.confusion_matrix(labelMatrixTest,predictedTest)))
        #savePredToFile(tweetMatrixTest,predictedTest,"predictedATest.json")
        saveTweetToCSVFile(predictedTest,"predicted.txt")
        #saveTweetToCSVFile(labelMatrixTest,"golden.txt")
        #f_importances(model.coef_,feature_names)
        #print(model._get_coef)



    
    
    