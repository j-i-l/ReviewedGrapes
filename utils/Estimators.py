from collections import defaultdict
from pyspark import keyword_only
from pyspark.ml import Estimator, Pipeline
from pyspark.ml.param.shared import (HasInputCol, HasOutputCol, Param, Params,
                                     HasInputCols, HasOutputCols)
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel
from pyspark.sql.functions import col

from reviewed_grapes.models import WordSetTrackerModel, MarginalMaximizerModel
from reviewed_grapes.params import HasWordSet, HasIndexWords


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
        self.__module__ = 'reviewed_grapes.utils.Estimators'

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

        def label_count_reduce(prev=None, row=None):
            label_count = prev.get('_label_count', defaultdict(int))
            label_count[row[input_col]] += 1
            prev['_label_count'] = label_count
            return prev

        counting = dataset.rdd.map(
            lambda row: {input_col: row[input_col]}
        ).reduce(
            lambda prev, row: label_count_reduce(prev, row)
        )
        label_count = counting['_label_count']
        labels, _counts = list(
            zip(*sorted(label_count.items(), key=lambda x: x[1], reverse=True))
        )
        top_label = labels[0]
        return MarginalMaximizerModel(inputCol=input_col, outputCol=output_col,
                                      topLabel=top_label)


class WordSetTracker(Estimator,
                     HasInputCol, HasOutputCol,
                     HasWordSet, HasIndexWords,
                     DefaultParamsReadable, DefaultParamsWritable):

    limitTo = Param(
        Params._dummy(),
        'limitTo', 'Maximal number of elements to use.'
    )

    @keyword_only
    def __init__(self, inputCol='words', outputCol='features',
                 limitTo=None, wordSet=None, indexWords=True):
        super().__init__()
        self._setDefault(limitTo=None)
        self._setDefault(indexWords=True)
        self._setDefault(wordSet=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol='words', outputCol='features',
                  limitTo=None, wordSet=None, indexWords=True):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def getLimitTo(self,):
        return self.getOrDefault(self.limitTo)

    def _fit(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        limit_to = self.getLimitTo()
        index_words = self.getIndexWords()
        word_set = self.getWordSet()

        limit = slice(None, limit_to)

        if not word_set:
            def all_words_count_map(row):
                """
                Counts the presence of a word.

                Multiple occurrences count only as one.
                """
                new_row = row.asDict()
                word_counts = defaultdict(int)
                for w in set(row[input_col]):
                    word_counts[w] += 1
                new_row['_word_count'] = word_counts
                return new_row

            def all_words_count(previous=None, row=None):
                """Reducing fct that counts the presence of words in the descriptions."""
                for w, c in row['_word_count'].items():
                    previous['_word_count'][w] += c
                return previous

            word_count = dataset.rdd.map(
                lambda row: all_words_count_map(row)
            ).reduce(lambda prev,
                     row: all_words_count(prev, row))['_word_count']
            if word_set is not None:
                word_count = {w: c
                              for w, c in word_count.items() if w in word_set}

            words, counts = list(zip(*sorted(word_count.items(),
                                             key=lambda x: x[1],
                                             reverse=True)))
            word_set_out = words[limit]
            word_count = counts[limit]
        else:
            word_set_out = word_set[limit]
            word_count = None

        return WordSetTrackerModel(inputCol=input_col,
                                   outputCol=output_col,
                                   wordSet=word_set_out,
                                   wordCount=word_count,
                                   indexWords=index_words)


class CompositeStringIndexer(Estimator,
                             HasInputCols, HasOutputCols,
                             DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        """
        String indexer over multiple columns.

        Returns a StringIndexerModel that is fitted over multiple columns
        and produces consistent indexing.
        This is in contrast to StringIndexer with the attribute `inputCols` that
        produces a distink indexer for each coloumn.

        Example:

        >>> testData = sc.parallelize([Row(id=0, label1="a", label2="b"),
        ...                            Row(id=1, label1="b", label2="c"),
        ...                            Row(id=2, label1="c", label2="e"),
        ...                            Row(id=3, label1="d", label2="f")], 3).toDF()
        >>> si = StringIndexer(inputCols=['label1', 'label2'], outputCols=['i1', 'i2'])
        >>> si.fit(testData).transform(testData).show()
        +---+------+------+---+---+
        | id|label1|label2| i1| i2|
        +---+------+------+---+---+
        |  0|     a|     b|0.0|0.0|
        |  1|     b|     c|1.0|1.0|
        |  2|     c|     e|2.0|2.0|
        |  3|     d|     f|3.0|3.0|
        +---+------+------+---+---+
        >>> csi = CompositeStringIndexer(inputCols=['label1', 'label2'],
        ...                              outputCols=['i1', 'i2'])
        >>> csi.fit(testData).transform(testData).show()
        +---+------+------+---+---+
        | id|label1|label2| i1| i2|
        +---+------+------+---+---+
        |  0|     a|     b|2.0|0.0|
        |  1|     b|     c|0.0|1.0|
        |  2|     c|     e|1.0|4.0|
        |  3|     d|     f|3.0|5.0|
        +---+------+------+---+---+

        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()

        combCol = 'combined'
        # We could also simply get the first column and then union the rest
        # no need for spark and sc in this case...
        # interim_schema = StructType([
        #     StructField(combCol, StringType(), True)
        # ])
        # interim_df = spark.createDataFrame(sc.emptyRDD(), interim_schema)
        interim_df = dataset.select(col(input_cols[0]).alias(combCol))
        if len(input_cols) > 1:
            for column in input_cols[1:]:
                interim_df = interim_df.union(
                    dataset.select(dataset[column].alias(combCol))
                )
        si = StringIndexer(inputCol=combCol, outputCol=f'{combCol}_indexed')
        sim = si.fit(interim_df)
        csim = StringIndexerModel()
        return csim.from_arrays_of_labels([sim.labels for _ in input_cols],
                                          inputCols=input_cols,
                                          outputCols=output_cols)


class CompositeOneHotEncoder(Estimator, HasInputCols, HasOutputCols,
                             DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        """
        Perform one hot encoding consistently accross multiple columns.

        Example:

        >>> testData = sc.parallelize([Row(id=0, label1="a", label2="b"),
        ...                            Row(id=1, label1="b", label2="c"),
        ...                            Row(id=2, label1="c", label2="e"),
        ...                            Row(id=3, label1="d", label2="f")], 3).toDF()
        >>> csi = CompositeStringIndexer(inputCols=['label1', 'label2'],
        ...                              outputCols=['i1', 'i2'])
        >>> compi_tD = csi.fit(testData).transform(testData)
        >>> ohenc = OneHotEncoder(inputCols=['i1', 'i2'], outputCols=['e1', 'e2'])
        >>> cohenc = CompositeOneHotEncoder(inputCols=['i1', 'i2'], outputCols=['e1', 'e2'])
        >>> ohenc.fit(compi_tD).transform(compi_tD).show()
        TODO:
        ADD OUTPUT HERE!
        >>> cohenc.fit(compi_tD).transform(compi_tD).show()
        +---+------+------+---+---+-------------+-------------+
        | id|label1|label2| i1| i2|           e1|           e2|
        +---+------+------+---+---+-------------+-------------+
        |  0|     a|     b|2.0|0.0|(5,[2],[1.0])|(5,[0],[1.0])|
        |  1|     b|     c|0.0|1.0|(5,[0],[1.0])|(5,[1],[1.0])|
        |  2|     c|     e|1.0|4.0|(5,[1],[1.0])|(5,[4],[1.0])|
        |  3|     d|     f|3.0|5.0|(5,[3],[1.0])|    (5,[],[])|
        +---+------+------+---+---+-------------+-------------+
        """
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def _fit(self, dataset):
        input_cols = self.getInputCols()
        output_cols = self.getOutputCols()

        combCol = 'combined'
        interim_df = dataset.select(col(input_cols[0]).alias(combCol))
        if len(input_cols) > 1:
            for column in input_cols[1:]:
                interim_df = interim_df.union(
                    dataset.select(dataset[column].alias(combCol))
                )
        ohenc = OneHotEncoder(inputCol=combCol, outputCol=f'{combCol}_enc', dropLast=False)
        ohencm = ohenc.fit(interim_df)
        ohencs = []
        for ic, oc in zip(input_cols, output_cols):
            ohencs.append(ohencm.copy())
            ohencs[-1].setInputCol(ic)
            ohencs[-1].setOutputCol(oc)
        del(ohenc)
        del(interim_df)
        return Pipeline(stages=ohencs).fit(dataset)
