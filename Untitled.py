#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import albumentations as alb

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,df,df_path,X_cols,y_cols,n_classes,input_shape,batch_size,augmentor = None ,shuffle = True,rescale = 1,aug_factor = 0):
        """
        df : the dataframe that contains the dataset paths
        
        X_cols : a list of columns names in df 
                 that will be the inputs of the model
                 
        y_cols : a list of columns names in df
                 that will be the outputs of the model
        
        input_shape : the shape of the input images
        
        batch_size : how many samples will be collected together as one batch
        
        shuffle : a boolean to shuffle the data samples or not
        
        rescale : a fraction representing the amount of rescaling to each image
        
        returns a data generator object
        """
        
        
        self.df = df.copy()
        self.df_path = df_path
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.rescale = rescale
        self.n = len(self.df)
        
        
        ###### Write your custom function here #######
        # example from :-
        # https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
        #self.n_name = df[y_cols['name']].nunique()
        #self.n_type = df[y_cols['type']].nunique()
        
        self.coords = df[y_cols[1:]]
        self.classes = df[y_cols[0]]
        self.n_classes = n_classes
        self.augmentation_factor = aug_factor
        self.augmentor = augmentor
    
    
    
    
    
    
    def __get_input(self, folder_path , path,positions , labels , index , target_size = None):
        images = []
        img_path = os.path.join(folder_path,path)
        image = tf.keras.preprocessing.image.load_img(img_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = image_arr.astype(np.float32) * self.rescale
        images.append(image_arr)
        
        
        if target_size != None:
            image_arr = tf.image.resize(image_arr,
                                        (target_size[0],
                                         target_size[1])).numpy()
        
        if self.augmentation_factor >0 :
            try: 
                for _ in range(self.augmentation_factor):
                    coords = self.df.loc[index , self.y_cols[1:]]
                    category = self.df.loc[index , self.y_cols[0]]
                    
                    augmented = self.augmentor(image=image_arr, bboxes=[coords], class_labels=[category])
                    
                    images.append(np.array(augmented['image']))

                    positions.append(np.array(augmented['bboxes'][0]))
                    l = tf.keras.utils.to_categorical(augmented['class_labels'][0],
                                                      num_classes = self.n_classes)
                    labels.append(np.array(l))
            
            except Exception as e:
                print(e)
        
        return images
    
    
    
    
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label,
                                             num_classes=num_classes)
                                
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        
        path_batch = batches[self.X_cols[0]]
        
        classes = batches[self.y_cols[0]]
        coords = batches[self.y_cols[1:]]
        
#         if self. == True:
#             target_size = self.input_shape[:-1]
        
        y0_list = [self.__get_output(y, self.n_classes) for y in classes] 
        y1_list = list(coords.values)
        
        
        
        list_of_images = [self.__get_input(folder_path = self.df_path,
                                           path = x,
                                           target_size = None,#self.input_shape,
                                           positions = y1_list ,
                                           labels=y0_list ,
                                           index = i) for i,x in enumerate(path_batch)]
        
        X_list = [item for sublist in list_of_images for item in sublist]
        
        
       
        X_batch = np.asarray(X_list)
        y0_batch = np.asarray(y0_list)
        y1_batch = np.asarray(y1_list)
        
        
        return X_batch, [y0_batch, y1_batch]
    
    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    
    def __len__(self):
        return self.n // self.batch_size

