from pyspark.ml.param.shared import Param, Params


class HasTopLabel(Params):
    topLabel = Param(Params._dummy(), 'topLabel',
                     'The most common label in the dataset')

    def __init__(self):
        super().__init__()

    def setTopLabel(self, value):
        return self._set(topLabel=value)

    def getTopLabel(self,):
        return self.getOrDefault(self.topLabel)


class HasWordSet(Params):
    wordSet = Param(Params._dummy(), 'wordSet',
                    'Set of words for which the presence is tracked')

    def __init__(self):
        super().__init__()

    def setWordSet(self, value):
        return self._set(wordSet=value)

    def getWordSet(self,):
        return self.getOrDefault(self.wordSet)


class HasWordCount(Params):
    wordCount = Param(
        Params._dummy(), 'wordCount',
        'Count of the number of occurences of a set of tracked words'
    )

    def __init__(self):
        super().__init__()

    def setWordCount(self, value):
        return self._set(wordCount=value)

    def getWordCount(self,):
        return self.getOrDefault(self.wordCount)


class HasIndexWords(Params):

    indexWords = Param(
        Params._dummy(),
        'indexWords',
        'Whether or not to return a list of words or a list of indices.')

    def __init__(self):
        super().__init__()

    def setIndexWords(self, value=False):
        return self._set(indexWords=bool(value))

    def getIndexWords(self,):
        return self.getOrDefault(self.indexWords)
