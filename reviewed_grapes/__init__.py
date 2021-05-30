import os
from functools import partial, update_wrapper
from .models import ReviewedGrapesModel

__all__ = ['MarginalModel',
           'CommonWordsModel',
           'SimilarWordsModel',
           'DissimilarWordsModel',
           'ExtremesWordsModel',
           'LowentropyWordsModel',
           ]
models_path = os.path.dirname(__file__)

# MarginalModel
marginal_m_path = os.path.join(models_path, 'fitted_models', 'MarginalModel')
MarginalModel = partial(ReviewedGrapesModel, modelPath=marginal_m_path)
update_wrapper(MarginalModel, ReviewedGrapesModel)

# CustomWordsModel
commonw_m_path = os.path.join(models_path, 'fitted_models',
                              'CommonWordsModel')
CommonWordsModel = partial(ReviewedGrapesModel, modelPath=commonw_m_path)
update_wrapper(CommonWordsModel, ReviewedGrapesModel)

# SimilarWordsModel
similarw_m_path = os.path.join(models_path, 'fitted_models',
                               'SimilarWordsModel')
SimilarWordsModel = partial(ReviewedGrapesModel, modelPath=similarw_m_path)
update_wrapper(SimilarWordsModel, ReviewedGrapesModel)

# DissimilarWordsModel
dissimilarw_m_path = os.path.join(models_path, 'fitted_models',
                                  'DissimilarWordsModel')
DissimilarWordsModel = partial(ReviewedGrapesModel,
                               modelPath=dissimilarw_m_path)
update_wrapper(DissimilarWordsModel, ReviewedGrapesModel)

# ExtremesWordsModel
extremesw_m_path = os.path.join(models_path, 'fitted_models',
                                'ExtremesWordsModel')
ExtremesWordsModel = partial(ReviewedGrapesModel, modelPath=extremesw_m_path)
update_wrapper(ExtremesWordsModel, ReviewedGrapesModel)

# LowentropyWordsModel
lowentropyw_m_path = os.path.join(models_path, 'fitted_models',
                                  'LowentropyWordsModel')
LowentropyWordsModel = partial(ReviewedGrapesModel,
                               modelPath=lowentropyw_m_path)
update_wrapper(LowentropyWordsModel, ReviewedGrapesModel)
