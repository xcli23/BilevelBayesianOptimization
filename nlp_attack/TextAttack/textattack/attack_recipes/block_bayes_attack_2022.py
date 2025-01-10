
"""
Attention + LSH

=======

(A Strong Baseline for Query Efficient Attacks in a Black Box Setting)

"""
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification
)
from textattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapHowNet
from textattack.goal_functions import UntargetedClassificationDiff
from textattack.search_methods import DiscreteBlockBayesAttack
from textattack import Attack
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import MaxWordsPerturbed

from .attack_recipe import AttackRecipe


class DiscreteBlockBayesAttackWordNet(AttackRecipe):
    """An implementation of the paper "A Strong Baseline for
    Query Efficient Attacks in a Black Box Setting", Maheshwary et al., 2021.

    The attack jointly leverages attention mechanism and locality sensitive hashing
    (LSH) to rank input words and reduce the number of queries required to attack
    target models. The attack iscevaluated on four different search spaces.

    https://arxiv.org/abs/2109.04775
    """
    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)


class DiscreteBlockBayesAttackHowNet(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)

class DiscreteBlockBayesAttackEmbedding(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassificationDiff(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)


class DiscreteBlockBayesAttackEmbeddingGen(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        transformation = WordSwapEmbedding(max_candidates=8)
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"question"}
        )
        constraints.append(input_column_modification)
        #
        # Language Model
        #
        #
        #
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassificationDiff(model)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        ) 
        return Attack(goal_function, constraints, transformation, search_method)



class DiscreteBlockBayesAttackWordNetHypothesis(AttackRecipe):
    """An implementation of the paper "A Strong Baseline for
    Query Efficient Attacks in a Black Box Setting", Maheshwary et al., 2021.

    The attack jointly leverages attention mechanism and locality sensitive hashing
    (LSH) to rank input words and reduce the number of queries required to attack
    target models. The attack iscevaluated on four different search spaces.

    https://arxiv.org/abs/2109.04775
    """

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)


class DiscreteBlockBayesAttackHowNetHypothesis(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)

class DiscreteBlockBayesAttackEmbeddingHypothesis(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassificationDiff(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)


class DiscreteBlockBayesAttackEmbeddingGenHypothesis(AttackRecipe):

    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        transformation = WordSwapEmbedding(max_candidates=8)
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        input_column_modification = InputColumnModification(
            ["question", "sentence"], {"sentence"}
        )
        constraints.append(input_column_modification)
        #
        # Language Model
        #
        #
        #
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassificationDiff(model)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)


from textattack.transformations import WordSwapMaskedLM
class DiscreteBlockBayesAttackBAE(AttackRecipe):
    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)

class DiscreteBlockBayesAttackBERTAttack(AttackRecipe):
    @staticmethod
    def build(model, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='v3', use_sod=True, dpp_type='dpp_posterior', max_loop=5, fit_iter=20, max_budget_key_type=''):
        print("block_size", block_size)
        print("batch_size", batch_size)
        print("update_step", update_step)
        print("max_patience", max_patience)
        print("post_opt", post_opt)
        print("use_sod", use_sod)
        print("dpp_type", dpp_type)
        print("max_loop", max_loop)
        print("fit_iter", fit_iter)
        print("max_budget_key_type", max_budget_key_type)
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=8)
        constraints = [RepeatModification(), StopwordModification()]

        use_constraint = UniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)

        goal_function = UntargetedClassificationDiff(model)
        search_method = DiscreteBlockBayesAttack(
            block_size=block_size,
            batch_size=batch_size,
            update_step=update_step,
            max_patience=max_patience,
            post_opt=post_opt,
            use_sod=use_sod,
            dpp_type=dpp_type, 
            max_loop=max_loop,
            fit_iter=fit_iter,
            max_budget_key_type=max_budget_key_type,
        )
        return Attack(goal_function, constraints, transformation, search_method)