import pickle
import math
import datetime
import inflection

import pandas  as pd
import numpy   as np

class Rossmann (object):
    def __init__(self):
        self.home_path ='parameters/'
        self.rescaling_year                           = pickle.load(open(self.home_path + 'rescaling_year_c01.pkl', 'rb'))
        self.rescaling_competition_distance           = pickle.load(open(self.home_path + 'rescaling_competition_distance_c01.pkl', 'rb'))
        self.rescaling_competition_open_since_month   = pickle.load(open(self.home_path + 'rescaling_competition_open_since_month_c01.pkl', 'rb'))
        self.rescaling_competition_open_since_year    = pickle.load(open(self.home_path + 'rescaling_competition_open_since_year_c01.pkl', 'rb'))
        self.rescaling_competition_open_timein_months = pickle.load(open(self.home_path + 'rescaling_competition_open_timein_months_c01.pkl', 'rb'))
        self.rescaling_promo2_since_timein_weeks      = pickle.load(open(self.home_path + 'rescaling_promo2_since_timein_weeks_c01.pkl', 'rb'))
        self.rescaling_promo2_since_year              = pickle.load(open(self.home_path + 'rescaling_promo2_since_year_c01.pkl', 'rb'))                                                                                                               
        self.transforming_store_type                  = pickle.load(open(self.home_path + 'transforming_store_type_c01.pkl', 'rb'))   
                                                                 

    def data_cleaning(self, df1):
        
        ## 1.1 RENAME COLUMNS
        
        old_cols = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
            'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)
        new_cols  = list(map(snakecase, old_cols))

        # rename
        df1.columns = new_cols
        df1.rename(columns={'promo': 'is_promo', 'promo2': 'promo2_participant', 'promo_interval':'promo2_interval'},  inplace = True)
        
                                                                 
        ## 1.2 FILLOUT NA

        # competition distance
        max_dist = df1['competition_distance'].max()
        new_max_dist = max_dist * 1.5
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: new_max_dist if math.isnan(x) else x)                                                        
        
        df1['date'] = pd.to_datetime(df1['date'])

        # 'competition_open_since_month' and 'competition_open_since_year'
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else 
                                                                  x['competition_open_since_month'], axis=1)

        df1['competition_open_since_year']  = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else 
                                                                  x['competition_open_since_year'], axis=1)
                                                                
        # promo2_since_week and 'promo2_since_year'
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else 
                                                       x['promo2_since_week'], axis=1)

        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else 
                                                       x['promo2_since_year'], axis=1)
        
        # 'promo_interval' 
        # creating a dictionary to replace the number of the month (which I will extract from date) with the name
        month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # replacing every NA in 'promo_inteval' with 0, because these are stores that did not participate in the promotion
        df1['promo2_interval'].fillna(0, inplace=True)

        # extracting the month from date with 'month_map'
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        # variable to tell if the purchase happened on a promotion
        df1['is_promo2'] = df1[['promo2_interval', 'month_map']].apply(lambda x: 0 if x['promo2_interval'] == 0 else 
                                                                                 1 if x['month_map'] in x['promo2_interval'].split(',') else 0, axis=1)
        
        ## 1.3 CHANGE VARIABLE TYPES
                                                                 
        # competiton
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(np.int64)
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype(np.int64)

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(np.int64)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(np.int64)  
      
        return df1
       
        
    def feature_engineering(self, df2):
        
        ## 2.1 FEATURE ENGINEERING    
                                                                 
        # year
        df2['year'] = df2['date'].dt.year.astype(np.int64)

        # month
        df2['month'] = df2['date'].dt.month.astype(np.int64)

        # day
        df2['day'] = df2['date'].dt.day.astype(np.int64)

        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(np.int64)

        # year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # creating 'competition_open_since' by combining 'competition_open_since_year' and 'competition_open_since_month'
        df2['competition_open_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                              month=x['competition_open_since_month'],day=1), axis=1)

        # creating 'competition_time_month' by calculating the difference between 'date' and 'competition_open_since'
        df2['competition_open_timein_months'] = ((pd.to_datetime(df2['date']).dt.date - pd.to_datetime(df2['competition_open_since']
                                                                                                      ).dt.date)/30
                                                ).apply(lambda x: x.days).astype(np.int64)

        # creating 'promo2_since' by 'combining promo2_since_year' and 'promo2_since_week'
        df2['promo2_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str) 
        df2['promo2_since'] = df2['promo2_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w'
                                                                                            ) - datetime.timedelta(days=7))  
        #the -1 and -%w pattern tells the parser to pick the monday in that week

        # creating 'promo2_since_timein_weeks' by calculating the difference between 'date' and 'promo2_since':
        df2['promo2_since_timein_weeks'] = ((pd.to_datetime(df2['date']).dt.date - pd.to_datetime(df2['promo2_since']).dt.date)/7
                                           ).apply(lambda x: x.days).astype(np.int64)

        # changing assortment attribute by given classification
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 
                                                              'extra' if x == 'b' else 
                                                              'extended')

        # changing 'state_holiday' attribute by given classification
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 
                                                                    'easter_holiday' if x == 'b' else 
                                                                    'christmas' if x == 'c' else 
                                                                    'regular_day')  
                                                                 
         ## 2.2 VARIABLE FILTERING
        
        df2 = df2[(df2['open']!= 0)]        
        cols_drop = ['open', 'promo2_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        return df2
    
    def data_preparation(self, df5):
        
        ## 3.1 RESCALING
        
        df5['year']                           = self.rescaling_year.transform(df5[['year']].values)
        df5['competition_distance']           = self.rescaling_competition_distance.transform(df5[['competition_distance']].values)                                                        
        df5['competition_open_since_month']   = self.rescaling_competition_open_since_month.transform(df5[['competition_open_since_month']].values)
        df5['competition_open_since_year']    = self.rescaling_competition_open_since_year.transform(df5[['competition_open_since_year']].values)
        df5['competition_open_timein_months'] = self.rescaling_competition_open_timein_months.transform(df5[['competition_open_timein_months']].values)
        df5['promo2_since_timein_weeks']      = self.rescaling_promo2_since_timein_weeks.transform(df5[['promo2_since_timein_weeks']].values)
        df5['promo2_since_year']              = self.rescaling_promo2_since_year.transform(df5[['promo2_since_year']].values)

        ## 3.2 ENCONDING
        
        # state_holiday - one hot enconding (each holiday becomes a column)
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - label enconding (each type becomes a value in a range)
        df5['store_type'] = self.transforming_store_type.transform(df5[['store_type']])

        # assortment - ordinal enconding (each assortment becomes a value in a hierarchy)
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)       
        
                                                                 
        ## 3.3 NATURE TRANSFORMATION

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi / 12)))

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52)))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi / 30)))

        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi / 7)))
        
        cols_selected = ['store', 'is_promo', 'is_promo2', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                         'competition_open_since_year', 'promo2_participant', 'promo2_since_week', 'promo2_since_year', 'competition_open_timein_months',
                         'promo2_since_timein_weeks', 'month_cos', 'week_of_year_sin', 'week_of_year_cos', 'day_sin', 'day_cos', 'day_of_week_sin', 'day_of_week_cos', 'year'] 
        
        return df5[cols_selected]

                                                                         
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)

        # joining prediction into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')