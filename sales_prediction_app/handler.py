import os
import pickle
import pandas as pd

from flask              import Flask, request, Response
from rossmann.Rossmann  import Rossmann 

# loading model
model = pickle.load(open('model/model_xgb_tuned_c01.pkl', 'rb'))

# initializing API
app = Flask(__name__)

@app.route('/rossmann/prediction', methods=['POST']) 
def rossmann_prediction():
    test_json = request.get_json()

    if test_json: # there is data
        if isinstance(test_json, dict): # if unique example
            test_df = pd.DataFrame(test_json, index=[0])
            print('1')
        else: # if multiple examples
            test_df = pd.DataFrame(test_json, columns=test_json[0].keys())
            print('2')

        # instantiate Rossmann Class
        pipeline = Rossmann()
        print('3')

        # data cleaning
        df1 = pipeline.data_cleaning(test_df)
        print('4')

        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        print('5')

        # data preparation
        df3 = pipeline.data_preparation(df2)
        print('6')

        # prediction
        df_response = pipeline.get_prediction(model, test_df, df3)
        print('7')

        return df_response

    else: # if empty
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('192.168.1.107', port=port)
