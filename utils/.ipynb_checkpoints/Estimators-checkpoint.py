from collections import defaultdict
from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import lit


class MarginalMaximizer(Estimator,
                        HasInputCol, HasOutputCol,
                        DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom estimator that predicts simply on the marginal probability of labels
    """

    @keyword_only
    def __init__(self, inputCol='label', outputCol='prediction'):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol='label', outputCol='prediction'):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def _fit(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        def label_count_reduce(previous=None, row=None):
            label_count = previous.get('_label_count', defaultdict(int))
            # variety_count = previous.get('_variety_count', defaultdict(int))
            label_count[row[input_col]] += 1
            # variety_count[row['variety']] += 1
            previous['_label_count'] = label_count
            # previous['_variety_count'] = variety_count
            return previous

        counting = dataset.rdd.map(
            lambda row: {input_col: row[input_col]}
        ).reduce(
            lambda prev, row: label_count_reduce(prev, row)
        )
        label_count = counting['_label_count']
        # variety_count = counting['_variety_count']
        # print(label_count, variety_count)
        labels, _counts = list(
            zip(*sorted(label_count.items(), key=lambda x: x[1], reverse=True))
        )
        top_label = labels[0]
        return MarginalMaximizerModel(inputCol=input_col, outputCol=output_col,
                                      topLabel=top_label)


class HasTopLabel(Params):
    topLabel = Param(Params._dummy(), 'topLabel',
                     'The most common label in the dataset')

    def __init__(self):
        super().__init__()

    def setTopLabel(self, value):
        return self._set(topLabel=value)

    def getTopLabel(self,):
        return self.getOrDefault(self.topLabel)


class MarginalMaximizerModel(Model, HasInputCol, HasOutputCol, HasTopLabel,
                             DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol, outputCol, topLabel):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol, outputCol, topLabel):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, database):
        output_col = self.getOutputCol()
        top_label = self.getTopLabel()
        return database.withColumn(output_col, lit(top_label))
