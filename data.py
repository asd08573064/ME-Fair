from torch.utils.data import Dataset
from data_augmentation import *
import torch
import os 
import io
import skimage

from skimage import io
from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import sampler
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def dataset_handler(dataset_name):
    num_class = {
        'isic2019' : 8,
        'fitzpatrick17k' : 114,
        'celebA' : 2,
        'prune_celebA' : 2,
        'utk' : 3
    }
    if dataset_name == 'isic2019':
        return ISIC2019
    if dataset_name == 'celebA':
        return celebA
    if dataset_name == 'prune_celebA':
        return prune_celebA
    if dataset_name == 'utk':
        return UTK

def get_weighted_sampler(df, label_level = 'low'):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[t] for t in df[label_level]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class ISIC2019_dataset_val(Dataset):
    def __init__(self, df=None, root_dir=None, transform=True):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], 'image']+'.jpg')
        image = io.imread(img_name)
        # some images have alpha channel, we just not ignore alpha channel
        if (image.shape[0] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        if self.transform:
            image = self.transform(image)
        label = self.df.loc[self.df.index[idx], 'low']
        gender = self.df.loc[self.df.index[idx], 'gender']

        return image, label, gender

class ISIC2019_val:
    def __init__(self, batch_size=64, add_trigger=False, model_name='VGG'): # 128
        self.batch_size = batch_size
        self.num_classes = 9
        if model_name == 'ResNet':
            self.image_size = 128
        else:
            self.image_size = 224 # for VGG

        predefined_root_dir = '/home/jinghao/ISIC_2019_Training_Input' # specify the image dir
        vali_df = pd.read_csv('/home/jinghao/Fairness/Shallow-Deep-Networks/isic2019_split/isic2019_val_pretraining.csv')
        test_df = pd.read_csv('/home/jinghao/Fairness/Shallow-Deep-Networks/isic2019_split/isic2019_test_pretraining.csv')
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.image_size).transforms
        vali_dataset= ISIC2019_dataset_val(df=vali_df, root_dir=predefined_root_dir, transform=test_transform)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= ISIC2019_dataset_val(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


        male_testset = ISIC2019_holdout_gender(test_df.copy(), "1")
        male_testset = ISIC2019_dataset_val(df=male_testset, root_dir=predefined_root_dir, transform=test_transform)
        male_testset = torch.utils.data.Subset(male_testset, torch.randperm(len(male_testset)))
        self.male_test_dataset_loader = torch.utils.data.DataLoader(male_testset, batch_size=64, shuffle=False, **kwargs)


        female_testset = ISIC2019_holdout_gender(test_df.copy(), "0")
        female_testset = ISIC2019_dataset_val(df=female_testset, root_dir=predefined_root_dir, transform=test_transform)
        female_testset = torch.utils.data.Subset(female_testset, torch.randperm(len(female_testset)))
        self.female_test_dataset_loader = torch.utils.data.DataLoader(female_testset, batch_size=64, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class ISIC2019_dataset_transform(Dataset):

    def __init__(self, df=None, root_dir=None, transform=True, feature_dict=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], 'image']+'.jpg')
        image = io.imread(img_name)
        # some images have alpha channel, we just not ignore alpha channel
        if (image.shape[0] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        if self.transform:
            image = self.transform(image)
        label = self.df.loc[self.df.index[idx], 'low']
        gender = self.df.loc[self.df.index[idx], 'gender']
        feature = {}
        if self.feature_dict != None:
            feature[0] = torch.Tensor(self.feature_dict[0][label])
            feature[1] = torch.Tensor(self.feature_dict[1][label])
            feature[2] = torch.Tensor(self.feature_dict[2][label])
            feature[3] = torch.Tensor(self.feature_dict[3][label])
            feature[4] = torch.squeeze(torch.Tensor(self.feature_dict[4][label]))
            return image, label, gender, feature
        else:
            return image, label, gender

def ISIC2019_holdout_gender(df, holdout_set: str = 'none'):
    if holdout_set == "0":
        remain_df = df[df.gender==1].reset_index(drop=True)
    elif holdout_set == "1":
        remain_df = df[df.gender==0].reset_index(drop=True)
    else:
        remain_df = df
    return remain_df

class ISIC2019:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, feature_dict=None):
        self.batch_size = batch_size
        self.num_classes = 9
        if model_name == 'ResNet':
            self.image_size = 128
        else:
            self.image_size = 224 # for VGG

        predefined_root_dir = '/home/jinghao/ISIC_2019_Training_Input' # specify the image dir
        train_df = pd.read_csv('/home/jinghao/Fairness/Shallow-Deep-Networks/isic2019_split/isic2019_train_pretraining.csv')
        vali_df = pd.read_csv('/home/jinghao/Fairness/Shallow-Deep-Networks/isic2019_split/isic2019_val_pretraining.csv')
        test_df = pd.read_csv('/home/jinghao/Fairness/Shallow-Deep-Networks/isic2019_split/isic2019_test_pretraining.csv')
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        sampler = get_weighted_sampler(train_df, label_level='low')
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=128, input_size=128, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=128, input_size=128, model_name=model_name).transforms
        aug_trainset =  ISIC2019_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_transform, feature_dict=feature_dict)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        train_dataset = ISIC2019_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_transform, feature_dict=feature_dict)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        vali_dataset= ISIC2019_dataset_transform(df=vali_df, root_dir=predefined_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= ISIC2019_dataset_transform(df=test_df, root_dir=predefined_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        male_trainset = ISIC2019_holdout_gender(train_df.copy(), "1")
        male_trainset= ISIC2019_dataset_transform(df=male_trainset, root_dir=predefined_root_dir, transform=test_transform)
        self.male_train_dataset = torch.utils.data.DataLoader(male_trainset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        female_trainset = ISIC2019_holdout_gender(train_df.copy(), "0")
        female_trainset= ISIC2019_dataset_transform(df=female_trainset, root_dir=predefined_root_dir, transform=test_transform)
        self.female_train_dataset = torch.utils.data.DataLoader(female_trainset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        male_testset = ISIC2019_holdout_gender(test_df.copy(), "1")
        male_testset= ISIC2019_dataset_transform(df=male_testset, root_dir=predefined_root_dir, transform=test_transform)
        self.male_test_dataset = torch.utils.data.DataLoader(male_testset, batch_size=self.batch_size, sampler=sampler, **kwargs)
        female_testset = ISIC2019_holdout_gender(test_df.copy(), "0")
        female_testset= ISIC2019_dataset_transform(df=female_testset, root_dir=predefined_root_dir, transform=test_transform)
        self.female_test_dataset = torch.utils.data.DataLoader(female_testset, batch_size=self.batch_size, sampler=sampler, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class celebA_dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, root_dir=None, transform=None, fairness_attribute=None, target_attribute=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.fairness_attribute = fairness_attribute
        self.target_attribute = target_attribute

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.loc[self.df.index[idx]]
        img_name = os.path.join(self.root_dir, row['image_id'])
        image = io.imread(img_name)
        if (image.shape[2] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        label = row[self.target_attribute]
        fairness_attribute = row[self.fairness_attribute]
        if self.transform:
            image = self.transform(image)

        return image, label, fairness_attribute

class celebA:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, fairness_attribute='Male', target_attribute='Bags_Under_Eyes', remove_img=None):
        self.batch_size = batch_size
        self.num_classes = 39

        predefined_root_dir_train = '/work/u6088529/datasets/train' # specify the image dir
        predefined_root_dir_test = '/work/u6088529/datasets/test' # specify the image dir
        predefined_root_dir_val = '//work/u6088529/datasets/val' # specify the image dir

        df_train_split=pd.read_csv('/home/u6088529/AAAI_2023/dataset/list_eval_partition.csv')
        df_attr=pd.read_csv('/home/u6088529/AAAI_2023/dataset/list_attr_celeba.csv')
        df_attr.head()
        df_attr.replace(-1,0,inplace=True)
        df_attr.head()
        train_df = df_attr[df_train_split['partition']==0]
        vali_df = df_attr[df_train_split['partition']==1]
        test_df = df_attr[df_train_split['partition']==2]
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

        if model_name == 'ResNet':
            self.image_size = 256
            self.crop_size = 224
        else:
            self.image_size = 256
            self.crop_size = 224
            
        sampler = get_weighted_sampler(train_df, label_level=fairness_attribute)
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        aug_trainset =  celebA_dataset(df=train_df, root_dir=predefined_root_dir_train, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size, **kwargs)
        train_dataset = celebA_dataset(df=train_df, root_dir=predefined_root_dir_train, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, **kwargs)
        vali_dataset= celebA_dataset(df=vali_df, root_dir=predefined_root_dir_val, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= celebA_dataset(df=test_df, root_dir=predefined_root_dir_test, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class prune_celebA_dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, root_dir=None, transform=None, fairness_attribute=None, target_attribute=None, remove_list=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.fairness_attribute = fairness_attribute
        self.target_attribute = target_attribute
        self.remove_list = remove_list

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.loc[self.df.index[idx]]
        img_name = os.path.join(self.root_dir, row['image_id'])
        try:
            image = io.imread(img_name)
        except:
            return None
        if (image.shape[2] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        label = row[self.target_attribute]
        fairness_attribute = row[self.fairness_attribute]
        if self.transform:
            image = self.transform(image)

        return image, label, fairness_attribute

class prune_celebA:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, fairness_attribute='Male', target_attribute='Attractive'):
        self.batch_size = batch_size
        self.num_classes = 39

        predefined_root_dir_train = '/home/u6088529/FSCL/datasets/train' # specify the image dir
        predefined_root_dir_test = '/home/u6088529/FSCL/datasets/prune_Attractive_Male' # specify the image dir
        predefined_root_dir_val = '/home/u6088529/FSCL/datasets/val' # specify the image dir

        df_train_split=pd.read_csv('./dataset/list_eval_partition.csv')
        df_attr=pd.read_csv('./dataset/list_attr_celeba.csv')
        df_attr.head()
        df_attr.replace(-1,0,inplace=True)
        df_attr.head()
        train_df = df_attr[df_train_split['partition']==0]
        vali_df = df_attr[df_train_split['partition']==1]
        test_df = df_attr[df_train_split['partition']==2]
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        with open("/home/u6088529/FSCL/prune_attractive_male.pkl", "rb") as fp:   # Unpickling
            remove_list = pickle.load(fp)

        if model_name == 'ResNet':
            self.image_size = 256
            self.crop_size = 224
        else:
            self.image_size = 256
            self.crop_size = 224
        # self.image_size = 128
        # self.crop_size = 112
        # print(test_df)
        for index, row in test_df.iterrows(): 
            if(test_df["image_id"][index] in remove_list):
                # print(test_df["image_id"][index])
                test_df.drop(index, inplace=True)
        # for index, row in test_df.iterrows(): 
        #     print(test_df["image_id"][index])

        
        sampler = get_weighted_sampler(train_df, label_level=fairness_attribute)
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        aug_trainset =  celebA_dataset(df=train_df, root_dir=predefined_root_dir_train, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size,**kwargs)
        train_dataset = celebA_dataset(df=train_df, root_dir=predefined_root_dir_train, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, **kwargs)
        vali_dataset= celebA_dataset(df=vali_df, root_dir=predefined_root_dir_val, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= celebA_dataset(df=test_df, root_dir=predefined_root_dir_test, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class utk_dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, root_dir=None, transform=None, fairness_attribute=None, target_attribute=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.fairness_attribute = fairness_attribute
        self.target_attribute = target_attribute

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.loc[self.df.index[idx]]
        img_name = os.path.join(self.root_dir, row['Name'])
        # print(img_name)
        image = io.imread(img_name)
        if (image.shape[2] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        # print(image.shape)
        if(self.fairness_attribute=='Race' and self.target_attribute=='Age'): # original MFD
            label = row[self.target_attribute]
            # print(label)
            if int(label) < 20:
                age_label = 0
            elif int(label) >= 20 and int(label) <= 40:
                age_label = 1
            else:
                age_label = 2
            fairness_att = row[self.fairness_attribute]
            # print(fairness_att)
            race_label = int(fairness_att)
            # if int(fairness_attribute) < 4:
            #     race_label = int(fairness_attribute)
            # else:
            #     race_label = 1
            if self.transform:
                image = self.transform(image)

            return image, age_label, race_label

        if(self.fairness_attribute=='Race' and self.target_attribute=='Gender'):
            label = row[self.target_attribute] # 0=male 1=female
            gender_label = label
            fairness_attribute = row[self.fairness_attribute]
            if int(fairness_attribute) == 0:
                race_label = 0
            else:
                race_label = 1

            if self.transform:
                image = self.transform(image)

            return image, gender_label, race_label

        if(self.fairness_attribute=='Age' and self.target_attribute=='Gender'):
            label = row[self.target_attribute]
            gender_label = label
            fairness_attribute = row[self.fairness_attribute]
            if int(fairness_attribute) < 35:
                age_label = 0
            else:
                age_label = 1

            if self.transform:
                image = self.transform(image)

            return image, gender_label, age_label
        # print(label)
        # print(fairness_attribute)
        # if self.transform:
        #     image = self.transform(image)

        # return image, age_label, race_label

class UTK:
    def __init__(self, batch_size=64, add_trigger=False, model_name=None, fairness_attribute='Race', target_attribute='Age', image_size=176, image_crop_size=176, remove_img=None): # 256, 224
        self.batch_size = batch_size
        self.num_classes = 39

        predefined_root_dir = '/work/u6088529/UTKFace' # specify the image dir
        df_train_split=pd.read_csv('/home/u6088529/AAAI_2023/extracted_info.csv')
        df_attr = pd.read_csv('/home/u6088529/AAAI_2023/extracted_info.csv')
        df_attr.head()
        df_attr.replace(-1,0,inplace=True)
        df_attr.head()
        train_df = df_attr
        data_count = np.zeros((4, 3), dtype=int)
        remain_count = np.zeros((4, 3), dtype=int)
        test_df = pd.DataFrame()
        race0_gender1_count = 2847
        race1_gender0_count = 5470
        for index, row in train_df.iterrows():
            test_drop_flag = 0
            sensitive_att = row[fairness_attribute]
            target_att = row[target_attribute]
            sens = 0
            target = 0
            if(fairness_attribute=='Race' and target_attribute=='Age'):
                if (sensitive_att < 4):
                    sens = sensitive_att
                else:
                    sens = -1
                    train_df.drop(index, inplace=True)
                    continue
                if (target_att < 20):
                    target = 0
                elif (target_att >= 20 and target_att <= 40):
                    target = 1
                else:
                    target = 2
                # if (sensitive_att == 0):
                #     sens = 0
                # else:
                #     sens = 1
                # if (target_att < 35):
                #     target = 0
                # else:
                #     target = 1
            elif(fairness_attribute=='Race' and target_attribute=='Gender'):
                if (sensitive_att == 0):
                    sens = 0
                else:
                    sens = 1
                target = target_att
            elif(fairness_attribute=='Age' and target_attribute=='Gender'):
                if (sensitive_att < 35):
                    sens = 0
                else:
                    sens = 1
                target = target_att
            data_count[sens, target] += 1
            remain_count[sens, target] += 1
            if data_count[sens, target] <= 100:
                # test_df = test_df.append(train_df.iloc[[index]],ignore_index=True)
                test_df = pd.concat([test_df, train_df.iloc[[index]]])
                train_df.drop(index, inplace=True)
                test_drop_flag = 1
            # For MFD Race checking
            if (sens == -1 ):
                train_df.drop(index, inplace=True)
            ## For FSCL pruning
            # if(fairness_attribute=='Race' and target_attribute=='Gender'):
            #     if(sens ==0 and target == 1 and race0_gender1_count > 0 and test_drop_flag == 0):
            #         race0_gender1_count -= 1
            #         train_df.drop(index, inplace=True)
            #         remain_count[sens, target] -= 1
            #     if(sens ==1 and target == 0 and race1_gender0_count > 0 and test_drop_flag == 0):
            #         race1_gender0_count -= 1
            #         train_df.drop(index, inplace=True)
            #         remain_count[sens, target] -= 1

        for i in range(4):
            for j in range(3):
                remain_count[i, j] -= 100
        # print(test_df)
        # print(train_df)
        print(remain_count)
        vali_df = test_df
        print(train_df)
        print(test_df)
        # test_df = test_df
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

        self.image_size = image_size
        self.crop_size = image_crop_size
        
            
        # sampler = get_weighted_sampler(train_df, label_level=fairness_attribute)
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.crop_size, model_name=model_name).transforms
        aug_trainset =  utk_dataset(df=train_df, root_dir=predefined_root_dir, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, batch_size=self.batch_size, **kwargs)
        train_dataset = utk_dataset(df=train_df, root_dir=predefined_root_dir, transform=train_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, **kwargs)
        vali_dataset= utk_dataset(df=vali_df, root_dir=predefined_root_dir, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= utk_dataset(df=test_df, root_dir=predefined_root_dir, transform=test_transform, fairness_attribute=fairness_attribute, target_attribute=target_attribute)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class CIFAR10:
    def __init__(self, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # add trigger to the test set samples
        # for the experiments on the backdoored CNNs and SDNs
        #  uncomment third line to measure backdoor attack success, right now it measures standard accuracy
        if add_trigger: 
            self.trigger_transform = transforms.Compose([AddTrigger(), transforms.ToTensor(), normalize])
            self.trigger_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.trigger_transform)
            self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

class CIFAR100:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset =  datasets.CIFAR100(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, batch_size=128):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000
        
        train_dir = 'data/tiny-imagenet-200/train'
        valid_dir = 'data/tiny-imagenet-200/val/images'

        normalize = transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262])
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)
        
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
