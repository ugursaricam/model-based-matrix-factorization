##########################################
# Model-Based Collaborative Filtering: Matrix Factorization
##########################################

# 1. Preparation of the Dataset
# 2. Modeling
# 3. Model Tuning
# 4. Final Model and Prediction

##########################################
# 1. Preparation of the Dataset
##########################################

import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how='left', on='movieId')

movie_ids = [130219, 356, 4422, 541]
movies = ['The Dark Knight (2011)',
          'Cries and Whispers (Viskningar och rop) (1972)',
          'Forrest Gump (1994)',
          'Blade Runner (1982)']

sample_df = df[df.movieId.isin(movie_ids)]

sample_df.shape # (97343, 6)

user_movie_df = sample_df.pivot_table(index=['userId'], columns=['title'], values='rating')
user_movie_df.head()

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)

##########################################
# 2. Modeling
##########################################

trainset, testset = train_test_split(data, test_size=.25)

svd_model = SVD()
svd_model.fit(trainset)

predictions = svd_model.test(testset)

accuracy.rmse(predictions) # 0.9329509453734651

sample_df[sample_df['userId'] == 1][['movieId', 'rating']]

#          movieId  rating
# 3612352      541     4.0

svd_model.predict(uid=1.0, iid=541, verbose=True) # actual: 4, model=4.013737589502537

sample_df[sample_df['userId'] == 137665.0][['movieId', 'rating']]

#           movieId  rating
# 3642707       541     5.0
# 14742596     4422     5.0

svd_model.predict(uid=137665.0, iid=541, verbose=True) # actual: 5, model=4.251685335248359
svd_model.predict(uid=137665.0, iid=4422, verbose=True) # actual: 5, model=4.6133415266795055

##########################################
# 3. Model Tuning
##########################################

param_grid = {'n_epochs': [5, 10, 20], 'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1, joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse'] # 0.9304491317273514
gs.best_params['rmse'] # {'n_epochs': 10, 'lr_all': 0.002}

gs.best_score['mae'] # 0.7161439481774107
gs.best_params['mae'] # {'n_epochs': 10, 'lr_all': 0.002}

##########################################
# 4. Final Model and Prediction
##########################################

dir(svd_model)
svd_model.n_epochs # 20

svd_model_final = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model_final.fit(data)

svd_model_final.predict(uid=1.0, iid=541, verbose=True) # actual: 4, model=4.238191144082312
svd_model_final.predict(uid=137665.0, iid=541, verbose=True) # actual: 5, model=4.246516571427077
svd_model_final.predict(uid=137665.0, iid=4422, verbose=True) # actual: 5, model=4.112607666248545


