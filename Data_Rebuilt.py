import os, argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings( 'ignore' )

        
PM2AWS = {
#    meta_pm   meta_aws
    '아름동': '세종고운',
    '신흥동': '세종연서',
    '노은동': '계룡',
    '문창동': '오월드',
    '읍내동': '장동',
    '정림동': '오월드',
    '공주': '공주',
    '논산': '논산',
    '대천2동': '대천항',
    '독곶리': '대산',
    '동문동': '태안',
    '모종동': '아산',
    '신방동': '성거',
    '예산군': '예산',
    '이원면': '태안',
    '홍성읍': '홍북',
    '성성동': '성거'
}

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def concat_train(path, dataset_path):
    TRAIN_list = os.listdir(os.path.join(dataset_path, 'TRAIN'))
    TRAIN_AWS_list = os.listdir(os.path.join(dataset_path, 'TRAIN_AWS'))
    Train_data = pd.DataFrame(columns=['연도', '일시', '측정소', 'PM2.5', '기온(°C)', '습도(%)'])

    for train in tqdm(TRAIN_list):
        # path
        train_path =  os.path.join(dataset_path, 'TRAIN', train)
        train_aws_path = os.path.join(dataset_path, 'TRAIN_AWS', f"{PM2AWS[train[:-4]]}.csv")

        # Interpolates of missing values.
        target = pd.read_csv(train_path).interpolate()
        target_aws = pd.read_csv(train_aws_path).interpolate()
        
        if target['PM2.5'].isnull().sum() > 0:
            # target['PM2.5'] = target['PM2.5'].fillna(method='ffill', inplace=True)
            target.fillna(target.mean(), inplace=True)
            #target['PM2.5'] = target['PM2.5'].interpolate()
            
        if target['PM2.5'].isnull().sum() > 0:
            print(train)
            null_rows = target.isna().sum(axis=1) > 0
            print(target[null_rows].index)

        # Save New data Files
        element = pd.concat([target, target_aws.loc[:,['기온(°C)','습도(%)']]], axis=1)
        element_path = os.path.join(path, 'TRAIN_DATA', train)
        element.to_csv(element_path, encoding='utf-8', index=False)
        
    print("Finish creating train data.")


def concat_test(path, dataset_path):
    TEST_list = os.listdir(os.path.join(dataset_path, 'TEST_INPUT'))
    TEST_AWS_list = os.listdir(os.path.join(dataset_path, 'TEST_AWS'))

    for test in tqdm(TEST_list):
        # path
        test_path =  os.path.join(dataset_path, 'TEST_INPUT', test)
        test_aws_path = os.path.join(dataset_path, 'TEST_AWS', f"{PM2AWS[test[:-4]]}.csv")

        # Interpolates of missing values.
        target = pd.read_csv(test_path)
        target_aws = pd.read_csv(test_aws_path).interpolate()
        
        # Save New data Files
        element = pd.concat([target, target_aws.loc[:,['기온(°C)','습도(%)']]], axis=1)
        element_path = os.path.join(path, 'TEST_DATA', test)
        element.to_csv(element_path, encoding='utf-8', index=False)
    
    print("Finish creating test data.")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=os.getcwd(), help='input your absolute path.')
    parser.add_argument('--dataset_path', type=str, default=os.path.join(os.getcwd(), 'dataset'), help='input your Dataset absolute path.')
    CFG = parser.parse_args()
    
    path = CFG.path
    dataset_path = CFG.dataset_path
    print(f"your absolute path: {path}")
    print(f"your Dataset absolute path: {dataset_path}")
    
    makedirs(os.path.join(path, 'TRAIN_DATA'))
    makedirs(os.path.join(path, 'TEST_DATA'))
    
    concat_train(path, dataset_path)
    concat_test(path, dataset_path)
    print("Exit the program.")
    

if __name__ == "__main__":
    main()