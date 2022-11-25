
import AutoClean as ac





def auto_clean(data_frame,encode_categ):
    
    
    print("Info before cleaning:")
    print("\n\nInfo: \n",data_frame.info())
    print("\n\nShape \n",data_frame.shape)
    print("\n\nTypes: \n",data_frame.dtypes)
    print("\n\nDescription: \n",data_frame.describe())
    
    print("\n\nCleaning data...")
    clean_pipeline = ac.AutoClean(data_frame,mode ="auto",verbose=True,encode_categ=encode_categ)
    clean_pipeline.output
    
    
    print("\n\n\n")
    print("\n\nInfo after cleaning: \n")
    print("\n\nInfo: \n",data_frame.info())
    print("\n\nShape \n",data_frame.shape)
    print("\n\nTypes: \n",data_frame.dtypes)
    print("\n\nDescription: \n",data_frame.describe())
   
   
    