#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import albumentations as alb

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,df,df_path,X_cols,y_cols,n_classes,input_shape,batch_size,shuffle = True,rescale = 1):
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
        
    
    def __get_input(self, folder_path , path,target_size = None):

        img_path = os.path.join(folder_path,path)
        image = tf.keras.preprocessing.image.load_img(img_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)*self.rescale
        
        if target_size != None:
            image_arr = tf.image.resize(image_arr,
                                        (target_size[0],
                                         target_size[1])).numpy()
        
        return image_arr
    
    
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
        
        X_batch = np.asarray([self.__get_input(self.df_path,x, self.input_shape) for x in path_batch])
        y0_batch = np.asarray([self.__get_output(y, self.n_classes) for y in classes])
        y1_batch = coords.values
        
        
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
    

