from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import lit, udf

from .params import HasWordSet, HasIndexWords, HasTopLabel, HasWordCount


class ReviewedGrapesModel(PipelineModel):

    def __new__(cls, inputCol='review', outputCol='prediction',
                modelPath=None):
        if modelPath is not None:
            self = cls.load(modelPath)
            # overwrite the input and output cols
            stages = self.stages
            s_first = stages[0]
            s_first.setInputCol(inputCol)
            s_last = stages[-1]
            s_last.setOutputCol(outputCol)
            self.stage = stages
        else:
            raise NotImplementedError(
                'For now only pretrained models are allowed, thus'
                ' the `modelPath` attribute must be set when instantiating'
                ' this class.'
            )

        return self


class MarginalMaximizerModel(Model, HasInputCol, HasOutputCol, HasTopLabel,
                             DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, topLabel=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, topLabel=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def _transform(self, database):
        output_col = self.getOutputCol()
        top_label = self.getTopLabel()
        return database.withColumn(output_col, lit(top_label))


class WordSetTrackerModel(Model, HasInputCol, HasOutputCol,
                          HasWordSet, HasWordCount,
                          HasIndexWords,
                          DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, wordSet=None,
                 wordCount=None, indexWords=True):
        """
        Indicates for a set of words the presence in the input column.
        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, wordSet=None,
                  wordCount=None, indexWords=True):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        word_set = self.getWordSet()
        index_words = self.getIndexWords()
        # word_count = self.getWordCount()

        # create the feature vector
        if index_words:

            @udf(returnType=VectorUDT())
            def toColumns(entry_words):  # row, stopwords):
                to_vec = [1 if word in entry_words else 0
                          for word in word_set]
                sparse_rep = {i: v for i, v in enumerate(to_vec) if v}
                return Vectors.sparse(len(to_vec), sparse_rep)
        else:

            @udf(returnType=ArrayType(StringType()))
            def toColumns(entry_words):  # row, stopwords):
                return list(word for word in word_set
                            if word in entry_words)

        return dataset.withColumn(output_col, toColumns(dataset[input_col]))
