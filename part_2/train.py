"""
code snippet for training lightGBM with a custom loss function
"""

import logging
import lightgbm as lgb

from custom_loss import AsymmetricLoss

# TODO: configure your model parameters
params = 
# TODO: define train and val datasets
train_lgb =
val_lgb = 

# instantiate custom loss class
al = AsymmetricLoss(penalty=3)

# train booster with custom loss functions
log.info("Training with Asymmetric Loss - Penalty: " + str(al.penalty))
bst = lgb.train(
    params,
    train_lgb,
    valid_sets=[val_lgb],
    fobj=al.custom_asymmetric_train,
    feval=al.custom_asymmetric_valid
)

              