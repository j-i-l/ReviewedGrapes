from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NLTKLemmatizer(Transformer, HasInputCol, HasOutputCol,
                     DefaultParamsReadable, DefaultParamsWritable):
    stopWords = Param(
        Params._dummy(),
        'stopWords', 'A list of stopwords to remove.'
    )
    onlyAlpha = Param(
        Params._dummy(),
        'onlyAlpha',
        'Whether or not to only keep words with normal letters'
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol='words', stopWords=None,
                 onlyAlpha=True):
        super().__init__()
        self._setDefault(stopWords=[])
        self._setDefault(onlyAlpha=True)
        self._setDefault(outputCol='words')
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol='words', stopWords=None,
                  onlyAlpha=True):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setStopWords(self, value):
        return self._set(stopWords=list(set(value)))

    def setOnlyAlpha(self, value):
        return self._set(onlyAlpha=bool(value))

    def getStopWords(self):
        return self.getOrDefault(self.stopWords)

    def getOnlyAlpha(self):
        return self.getOrDefault(self.onlyAlpha)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def _transform(self, dataset):
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        stop_words = self.getStopWords()
        only_alpha = self.getOnlyAlpha()

        @udf(returnType=ArrayType(StringType()))
        def lemmatize_words(text):
            """Adding a column with an set of lemmatized words."""
            tokenized = word_tokenize(text)
            wnl = WordNetLemmatizer()
            lemmatized = [wnl.lemmatize(w, pos='v') for w in tokenized
                          if w not in stop_words]
            if only_alpha:
                lemmatized = [w for w in lemmatized
                              if all(c.isalpha() for c in w)]
            return list(set(lemmatized))

        return dataset.withColumn(output_col,
                                  lemmatize_words(dataset[input_col]))
